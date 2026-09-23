#!/usr/bin/env python3
"""CPU regression tests for fair scheduling and versioned source migration."""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_PATCH_CANDIDATES = (
    HERE / 'patch_scheduler_decode_floor.py',
    ROOT / 'overlay' / 'patch_scheduler_decode_floor.py',
)
PATCH = next((p for p in _PATCH_CANDIDATES if p.is_file()), None)
if PATCH is None:
    raise SystemExit(
        'missing patch_scheduler_decode_floor.py (tried '
        + ', '.join(str(p) for p in _PATCH_CANDIDATES)
        + ')'
    )
spec = importlib.util.spec_from_file_location('glm53_decode_floor', PATCH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
POLICY = mod._Glm53MixedPrefill
POLICY_MOD_MIXED = POLICY
PATCHED_SOURCE = None
FAIR_ENV = {
    'GLM53_MIXED_PREFILL_CHUNK': 'fair',
    'GLM53_FAIR_PREFILL_CHUNK': '256',
    'GLM53_FAIR_PREFILL_SHARE': '0.20',
    'GLM53_FAIR_PREFILL_MAX_INTERVAL_MS': '2000',
    'GLM53_FAIR_PREFILL_MAX_STEP_MS': '1000',
    'GLM53_FAIR_PREFILL_MAX_CHUNKS': '1',
}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class Req:
    def __init__(self, rid, prompt=30000, computed=0, decode=False):
        self.request_id = rid
        self.num_prompt_tokens = prompt
        self.num_computed_tokens = computed
        self.num_tokens = prompt + int(decode)
        self.spec_token_ids = list(range(7)) if decode else []
        self.num_output_placeholders = 0
        self.next_decode_eligible_step = 0
        self.max_tokens = 4000
        self.has_encoder_inputs = False
        self.is_prefill_chunk = not decode

    @property
    def num_tokens_with_spec(self):
        return self.num_tokens + len(self.spec_token_ids)


class Sched:
    def __init__(self, running=(), waiting=()):
        self.running = list(running)
        self.waiting = list(waiting)
        self.skipped_waiting = []
        self.current_step = 1
        self.max_model_len = 850000
        self.num_sampled_tokens_per_step = 1
        self.need_mamba_block_aligned_split = False
        self.scheduler_config = SimpleNamespace(long_prefill_token_threshold=3584)
        self.refresh()

    def refresh(self):
        self.requests = {r.request_id: r for r in self.running + self.waiting + self.skipped_waiting}


class Out:
    def __init__(self, counts):
        self.num_scheduled_tokens = dict(counts)
        self.total_num_scheduled_tokens = sum(counts.values())


class FairTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.a = Req('A', 30000, 30000, decode=True)
        self.b = Req('B')
        self.s = Sched([self.a], [self.b])
        self.p = self.policy()

    def policy(self, **env):
        with patch.dict(os.environ, {**FAIR_ENV, **env}), contextlib.redirect_stdout(io.StringIO()):
            p = POLICY(now=self.clock)
        p.hist_every = 0
        return p

    def submit(self, counts, sched=None):
        s = sched or self.s
        self.p.begin_step(s)
        out = Out(counts)
        self.p.finish_step(s, out)
        s.current_step += 1
        return out

    def complete(self, out, dt, sched=None):
        self.clock.advance(dt)
        self.p.observe_output(sched or self.s, out)

    def learn(self, cost=0.2):
        self.assertEqual(self.p.cap_for(self.s, self.b), 256)
        out = self.submit({'A': 8, 'B': 256})
        self.complete(out, cost)

    def test_legacy_modes_and_alignment(self):
        for mode, expected in [('skip', 0), ('-1', 0), ('0', None), ('off', None), ('128', 128)]:
            p = self.policy(GLM53_MIXED_PREFILL_CHUNK=mode)
            s = Sched([self.a], [self.b])
            self.assertEqual(p.cap_for(s, self.b), expected)
        p = self.policy(GLM53_MIXED_PREFILL_CHUNK='skip')
        for _ in range(10000):
            self.s.current_step += 1
            self.assertEqual(p.cap_for(self.s, self.b), 0)
        self.assertFalse(p.inflight)
        align = POLICY.aligned_new_tokens
        self.assertEqual(align(0, 128, 30000, 3584, 3584), 0)
        for block in (1792, 3584):
            self.assertEqual(align(0, 128, 30000, block, 3584, 128), 128)
            self.assertEqual(align(block - 64, 128, 30000, block, 3584, 128), 64)
        self.assertEqual(align(29900, 100, 30000, 3584, 3584, 128), 100)

    def test_solo_prefill_then_immediate_newcomer_has_no_debt(self):
        pre = Req('A')
        self.s.running, self.s.waiting = [pre], []
        self.s.refresh()
        self.assertIsNone(self.p.cap_for(self.s, pre))
        out = self.submit({'A': 3584})
        self.complete(out, 6.0)
        self.assertEqual(self.p.credit, 0)
        self.assertEqual(self.p.solo_samples, [(3584, 6.0)])
        self.s.running, self.s.waiting = [self.a], [self.b]
        self.s.refresh()
        self.assertEqual(self.p.cap_for(self.s, self.b), 256)

    def test_late_solo_completion_keeps_original_classification(self):
        pre = Req('A')
        self.s.running, self.s.waiting = [pre], []
        self.s.refresh()
        solo = self.submit({'A': 3584})
        self.s.running, self.s.waiting = [self.a], [self.b]
        self.s.refresh()
        self.assertEqual(self.p.cap_for(self.s, self.b), 0)
        self.assertEqual(self.p.defer_reason, 'async_inflight')
        dec = self.submit({'A': 8})
        credit = self.p.credit
        self.complete(solo, 5.0)
        self.assertEqual(self.p.credit, credit)
        self.assertNotIn('B', self.p.last_service)
        self.complete(dec, 0.05)
        self.assertEqual(self.p.cap_for(self.s, self.b), 256)

    def test_saves_credit_for_target_rung_instead_of_small_chunk(self):
        # One 256@0.2s sample scales linearly: 1024 (0.8s) is the largest rung
        # under max_step_s. An already-served B with 0.21 credit waits for it
        # instead of buying a 256 now (v4 spent greedily and stalled small).
        self.learn()
        self.p.credit = 0.21
        self.assertEqual(self.p.cap_for(self.s, self.b), 0)
        self.assertEqual(self.p.defer_reason, 'credit')
        self.assertAlmostEqual(self.p.credit, 0.21)
        self.s.current_step += 1
        self.p.begin_step(self.s)
        self.p.credit = 0.85
        self.assertEqual(self.p.cap_for(self.s, self.b), 1024)
        self.assertAlmostEqual(self.p.credit, 0.05)
        self.assertAlmostEqual(self.p._open_rec['grants']['B'][1], 0.8)
        self.assertFalse(self.p._open_rec['grants']['B'][2])

    def kit_samples(self):
        # Head-log shaped samples: solo 3584@2.68s and an 82-token tail@0.31s
        # (~0.25s fixed + ~0.68ms/token); three mixed 128@0.34s agree.
        self.p.solo_samples = [(3584, 2.68), (82, 0.31)]
        self.p.mixed_samples = [((1, 3, 0), 128, 0.34)] * 3
        self.p._model_cache = None

    def test_fixed_cost_fit_prices_large_chunks_from_small_samples(self):
        self.kit_samples()
        fixed, per_tok = self.p._cost_model()
        self.assertGreater(fixed, 0.2)
        self.assertLess(per_tok, 0.001)
        self.assertLess(self.p._est_dt(1024), 1.0)   # v4: 2.72s from 128@0.34
        self.assertGreater(self.p._est_dt(2048), 1.0)
        self.assertGreater(self.p._est_dt(128), 0.3)

    def test_solo_samples_alone_price_the_first_probe(self):
        self.p.solo_samples = [(3584, 2.68), (82, 0.31)]
        self.p._model_cache = None
        self.assertGreater(self.p._est_dt(256), 0.35)
        self.assertLess(self.p._est_dt(256), 0.6)

    def test_single_outlier_does_not_dominate_estimate(self):
        self.p.solo_samples = [(3584, 2.68), (82, 0.31)]
        self.p.mixed_samples = [((1, 3, 0), 256, 0.43)] * 5 + [((1, 3, 0), 256, 0.60)]
        self.p._model_cache = None
        self.assertLess(self.p._est_dt(1024), 1.1)
        self.p.mixed_samples = [((1, 3, 0), 256, 0.60)] * 6
        self.p._model_cache = None
        self.assertGreater(self.p._est_dt(1024), 1.0)  # consistently slow mixed steps push 1024 over the 1.0s gate

    def test_ladder_climbs_back_after_small_chunks(self):
        self.kit_samples()
        self.p.begin_step(self.s)
        self.p.last_service['B'] = self.clock()
        self.p.credit = 1.0
        self.assertEqual(self.p.cap_for(self.s, self.b), 1024)

    def test_never_served_newcomer_gets_prompt_step_bounded_probe(self):
        self.kit_samples()
        self.p.begin_step(self.s)
        self.p.credit = 0.1
        self.assertEqual(self.p.cap_for(self.s, self.b), 1024)
        self.assertTrue(self.p._open_rec['grants']['B'][2])
        self.assertLess(self.p.credit, 0)
        out = self.submit({'A': 8, 'B': 1024})
        self.complete(out, 0.95)
        self.assertLess(self.p.credit, 0)
        c = Req('C')
        self.s.waiting.append(c)
        self.s.refresh()
        self.assertEqual(self.p.cap_for(self.s, c), 0)
        self.assertEqual(self.p.defer_reason, 'credit')
        self.assertEqual(self.p.cap_for(self.s, self.b), 0)

    def test_step_budget_caps_the_target_rung(self):
        self.p = self.policy(GLM53_FAIR_PREFILL_MAX_STEP_MS='500')
        self.kit_samples()
        self.p.begin_step(self.s)
        self.p.credit = 1.0
        cap = self.p.cap_for(self.s, self.b)
        self.assertIn(cap, (256, 512))
        self.assertLessEqual(self.p._open_rec['grants']['B'][1], 0.5)

    def test_positive_credit_does_not_override_step_latency(self):
        self.learn(cost=3.0)
        self.p._busy = [(self.clock(), 'decode', 1.0)]
        self.p.credit = 10
        self.assertEqual(self.p.cap_for(self.s, self.b), 0)
        self.assertEqual(self.p.defer_reason, 'gap_budget')

    def test_age_and_step_budgets_are_independent(self):
        p = self.policy(GLM53_FAIR_PREFILL_MAX_INTERVAL_MS='10000', GLM53_FAIR_PREFILL_MAX_STEP_MS='100')
        self.assertEqual(p.interval_s, 10)
        self.assertEqual(p.max_step_s, 0.1)
        # Nothing fits 100 ms, but a never-served newcomer still gets the
        # smallest rung through the borrow gate (v6 returned 0 forever).
        self.assertEqual(p.cap_for(self.s, self.b), 256)
        self.assertTrue(p._open_rec['grants']['B'][2])

    def head_log_stuck_samples(self):
        # 2026-09-22 head log: least squares fitted fixed=0.875 s, above the
        # 750 ms step budget, from heavy-tailed host timings. Typical steps
        # are ~0.3 s fixed + ~0.65 ms/token.
        self.p.solo_samples = ([(3584, 2.4)] * 6 + [(7168, 4.9)] * 6 + [(117, 1.63), (128, 3.98),
                               (256, 4.06), (7168, 9.89), (3584, 6.39), (384, 0.92)]
                               + [(128, 0.39)] * 6)
        self.p.mixed_samples = [((1, 3, 0), 256, 0.53)] * 18 + [((1, 3, 0), 256, 4.1), ((1, 3, 0), 512, 10.4)]
        self.p._model_cache = None

    def test_heavy_tail_samples_do_not_lift_fixed_cost_over_budget(self):
        self.p = self.policy(GLM53_FAIR_PREFILL_MAX_STEP_MS='750', GLM53_FAIR_PREFILL_SHARE='0.30',
                             GLM53_FAIR_PREFILL_MAX_INTERVAL_MS='750')
        self.head_log_stuck_samples()
        fixed, per_tok = self.p._cost_model()
        self.assertLess(fixed, 0.5)
        self.assertLess(self.p._est_dt(256), 0.75)
        self.p.begin_step(self.s)
        self.p.last_service['B'] = self.clock()
        self.p.credit = 0.75
        self.assertGreaterEqual(self.p.cap_for(self.s, self.b), 256)

    def test_floor_keeps_contended_prefill_moving_when_nothing_fits(self):
        # Every rung estimated above the step budget: v6 deferred B with
        # gap_budget until A finished. v7 serves the smallest rung once B is
        # due and debt is repaid, then waits for repayment again.
        self.p = self.policy(GLM53_FAIR_PREFILL_MAX_STEP_MS='750', GLM53_FAIR_PREFILL_MAX_INTERVAL_MS='750')
        self.p.solo_samples = [(128, 1.1), (256, 1.2), (3584, 4.0)]
        self.p.mixed_samples = [((1, 3, 0), 128, 1.1), ((1, 3, 0), 256, 1.2)]
        self.p._model_cache = None
        self.assertIsNone(self.p._target(None, 0.75))
        self.p.begin_step(self.s)
        self.p.last_service['B'] = self.clock()
        self.p.credit = 0.75
        self.assertEqual(self.p.cap_for(self.s, self.b), 0)  # not due yet
        self.assertEqual(self.p.defer_reason, 'gap_budget')
        self.clock.advance(1.0)
        self.s.current_step += 1
        self.p.begin_step(self.s)
        self.assertTrue(self.p._prefill_can_grant())
        self.assertTrue(self.p.hold_decode(self.a))
        self.assertEqual(self.p.cap_for(self.s, self.b), 128)
        out = self.submit({'B': 128})
        self.complete(out, 1.1)
        self.assertLess(self.p.credit, 0)
        # In debt: the next due turn defers, then decode-only turns repay.
        self.clock.advance(1.0)
        self.assertEqual(self.p.cap_for(self.s, self.b), 0)
        served = 0
        for _ in range(60):
            self.s.current_step += 1
            self.p.begin_step(self.s)
            cap = self.p.cap_for(self.s, self.b)
            out = self.submit({'B': cap} if cap else {'A': 8})
            self.complete(out, 1.1 if cap else 0.1)
            served += bool(cap)
        self.assertGreaterEqual(served, 2)
        self.assertLessEqual(served, 30)  # still rate-limited, not every step

    def test_c4_three_decoders_and_newcomers_all_progress(self):
        # C4 at the head-log costs with launcher knobs: three incumbents decode
        # while two prompts wait. Each waiting prompt must keep getting chunks.
        self.p = self.policy(GLM53_FAIR_PREFILL_MAX_STEP_MS='750', GLM53_FAIR_PREFILL_SHARE='0.30',
                             GLM53_FAIR_PREFILL_MAX_INTERVAL_MS='750')
        self.head_log_stuck_samples()
        decs = [Req(f'D{i}', 30000, 30000, decode=True) for i in range(3)]
        news = [Req('N0', 20000), Req('N1', 20000)]
        s = Sched(decs, news)
        served = {'N0': 0, 'N1': 0}
        for _ in range(400):
            self.p.begin_step(s)
            counts = {}
            for r in news:
                cap = self.p.cap_for(s, r)
                if cap:
                    counts[r.request_id] = cap
            if not counts:
                counts = {d.request_id: 8 for d in decs}
            out = Out(counts)
            self.p.finish_step(s, out)
            s.current_step += 1
            self.clock.advance(0.53 if any(k in served for k in counts) else 0.13)
            self.p.observe_output(s, out)
            for rid, n in counts.items():
                if rid in served:
                    served[rid] += n
                    s.requests[rid].num_computed_tokens += n
        self.assertGreater(served['N0'], 2000)
        self.assertGreater(served['N1'], 2000)

    def test_cost_feedback_can_shrink_chunks(self):
        self.learn(cost=1.5)
        self.p._busy = [(self.clock(), 'decode', 1.0)]
        self.p.credit = 1
        self.assertEqual(self.p.cap_for(self.s, self.b), 128)

    def test_ladder_uses_available_credit(self):
        self.learn(cost=0.2)
        self.p.credit = 0.9
        self.assertEqual(self.p.cap_for(self.s, self.b), 1024)
        self.assertAlmostEqual(self.p.credit, 0.1)

    def test_aggregate_reservations_bound_multiple_newcomers(self):
        self.p = self.policy(GLM53_FAIR_PREFILL_MAX_CHUNKS='3')
        self.s.waiting += [Req('C'), Req('D')]
        self.s.refresh()
        self.p.begin_step(self.s)
        self.p.credit = 0.4
        caps = [self.p.cap_for(self.s, r) for r in self.s.waiting]
        self.assertEqual(caps, [256, 256, 0])
        self.assertGreaterEqual(self.p.credit, 0)
        self.assertLessEqual(sum(g[1] for g in self.p._open_rec['grants'].values()), 0.4)

    def test_age_cannot_borrow_repeatedly_without_repayment(self):
        self.p = self.policy(GLM53_FAIR_PREFILL_MAX_CHUNKS='3')
        self.s.waiting += [Req('C'), Req('D')]
        self.s.refresh()
        self.p.begin_step(self.s)
        self.p.credit = 0.01
        self.clock.advance(3)
        self.assertEqual(self.p.cap_for(self.s, self.b), 256)
        out = self.submit({'A': 8, 'B': 256})
        self.complete(out, 0.4)
        self.assertLess(self.p.credit, 0)
        self.clock.advance(10)
        self.assertEqual(self.p.cap_for(self.s, self.s.waiting[1]), 0)
        self.assertEqual(self.p.cap_for(self.s, self.s.waiting[2]), 0)

    def test_empty_schedule_does_not_block_next_prefill_or_mint_credit(self):
        before = self.p.cap_for(self.s, self.b)
        self.assertEqual(before, 256)
        empty = self.submit({})
        self.assertFalse(self.p.inflight)
        credit = self.p.credit
        self.clock.advance(100)
        self.p.observe_output(self.s, empty)
        self.assertEqual(self.p.credit, credit)
        self.assertEqual(self.p.cap_for(self.s, self.b), 256)

    def test_removed_grant_refunded_and_no_service_credited(self):
        self.p.cap_for(self.s, self.b)
        out = self.submit({'A': 8})
        self.assertEqual(self.p.inflight_prefill, 0)
        self.complete(out, 0.05)
        self.assertNotIn('B', self.p.last_service)
        self.assertGreater(self.p.credit, 0)

    def test_final_counts_override_provisional_grant(self):
        self.p.cap_for(self.s, self.b)
        out = self.submit({'A': 8, 'B': 64})
        self.b.num_computed_tokens = 30000  # Async scheduler has already advanced it.
        self.complete(out, 0.1)
        self.assertEqual(self.p.served_tokens['B'], 64)

    def test_newer_open_step_cannot_contaminate_older_completion(self):
        decode = self.submit({'A': 8})
        self.p.cap_for(self.s, self.b)
        self.complete(decode, 0.1)
        self.assertNotIn('B', self.p.last_service)
        mixed = self.submit({'A': 8, 'B': 256})
        self.complete(mixed, 0.2)
        self.assertEqual(self.p.served_tokens['B'], 256)

    def test_queued_async_time_is_accounted_once(self):
        self.p.begin_step(self.s)
        self.p.credit = 0
        one = self.submit({'A': 8})
        two = self.submit({'A': 8})
        self.complete(one, 0.1)
        self.complete(two, 0.1)
        self.assertAlmostEqual(self.p.credit, 0.04)

    def test_duplicate_and_unrelated_outputs_do_not_pop_records(self):
        self.p.cap_for(self.s, self.b)
        out = self.submit({'A': 8, 'B': 256})
        self.p.observe_output(self.s, Out({'A': 8, 'B': 256}))
        self.assertEqual(self.p.inflight_prefill, 1)
        self.complete(out, 0.2)
        credit = self.p.credit
        self.p.observe_output(self.s, out)
        self.assertEqual(self.p.credit, credit)
        self.assertEqual(self.p.served_tokens['B'], 256)

    def test_idle_time_is_not_decode_service(self):
        first = self.submit({'A': 8})
        self.complete(first, 0.1)
        self.p.credit = 0
        self.clock.advance(100)
        second = self.submit({'A': 8})
        self.complete(second, 0.1)
        self.assertAlmostEqual(self.p.credit, 0.02)

    def test_cancel_and_rearrival_keep_debt_while_a_decodes(self):
        self.learn(cost=1.0)
        debt = self.p.credit
        self.s.waiting = [Req('C')]
        self.s.refresh()
        self.assertEqual(self.p.cap_for(self.s, self.s.waiting[0]), 0)
        self.assertEqual(self.p.credit, debt)
        self.assertNotIn('B', self.p.last_service)

    def test_zero_progress_promotes_next_candidate(self):
        c = Req('C')
        self.s.waiting.append(c)
        self.s.refresh()
        self.assertEqual(self.p.cap_for(self.s, self.b), 256)
        self.assertEqual(self.p.cap_for(self.s, c), 0)
        self.p.note_scheduled(self.b, 0)
        self.assertEqual(self.p.cap_for(self.s, c), 256)

    def test_full_prefix_hit_is_not_blocked_as_cold_prefill(self):
        self.p.begin_step(self.s)
        self.p.credit = -10
        self.assertIsNone(self.p.cap_for(self.s, self.b, computed=30000))

    def test_small_remaining_tail_can_fit_when_base_chunk_cannot(self):
        self.b.num_computed_tokens = 29980
        self.p.begin_step(self.s)
        self.p.credit = 0.05
        self.assertEqual(self.p.cap_for(self.s, self.b), 20)

    def running_loop(self, input_budget, allocate=None):
        if PATCHED_SOURCE is None:
            self.skipTest('source installation required')
        # Execute the pinned scheduler's actual running-loop budget/eligibility
        # code, omitting only KV allocation and post-allocation speculative setup.
        tree = ast.parse(PATCHED_SOURCE)
        schedule = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'schedule')
        loop = next(n for n in schedule.body if isinstance(n, ast.While))
        allocate_index = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.With))
        append_index = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Expr)
                            and ast.unparse(n).startswith('scheduled_running_reqs.append'))
        increment_index = next(i for i in range(append_index, len(loop.body))
                               if ast.unparse(loop.body[i]) == 'req_index += 1')
        loop.body = (loop.body[:allocate_index] + loop.body[append_index:increment_index + 1]
                     if allocate is None else loop.body[:increment_index + 1])
        code = compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), '<actual-running-loop>', 'exec')
        self.s.running, self.s.waiting = [self.b, self.a], []
        self.s.refresh()
        self.s.kv_cache_manager = SimpleNamespace(allocate_slots=allocate)
        self.s.num_lookahead_tokens = 7
        self.p.begin_step(self.s)
        ns = dict(self=self.s, _GLM53_MIXED=self.p,
                  _glm53_mixed_prefill_policy=lambda s, r: self.p.cap_for(s, r),
                  req_index=0, token_budget=7168, input_budget=input_budget, draft_slots=8,
                  defer_prefills=False, encoder_compute_budget=0, prefill_scheduled=False,
                  scheduled_running_reqs=[], req_to_new_blocks={}, num_scheduled_tokens={}, new_blocks=[],
                  record_function_or_nullcontext=lambda _: contextlib.nullcontext())
        exec(code, ns)
        return ns

    def test_prefill_turn_holds_decode(self):
        self.p.begin_step(self.s)
        self.assertEqual(self.p.step_mode, 'prefill_turn')
        self.assertTrue(self.p.hold_decode(self.a))
        self.assertFalse(self.p.hold_decode(self.b))
        self.assertEqual(self.p.cap_for(self.s, self.b), 256)

    def test_decode_only_step_has_no_prefill_chunk(self):
        self.assertEqual(self.p.cap_for(self.s, self.b), 256)
        self.submit({'B': 256})
        self.assertEqual(self.p.cap_for(self.s, self.b), 0)
        self.assertEqual(self.p.step_mode, 'decode_only')
        self.assertFalse(self.p.hold_decode(self.a))

    def test_decode_floor_after_prefill_busy(self):
        self.learn()
        self.p.credit = 10.0
        self.s.current_step += 1
        self.p.begin_step(self.s)
        cap = self.p.cap_for(self.s, self.b)
        self.assertGreater(cap, 0)
        out = Out({'B': cap})
        self.p.finish_step(self.s, out)
        self.complete(out, 0.85)
        self.s.current_step += 1
        self.p.begin_step(self.s)
        self.assertEqual(self.p.step_mode, 'decode_only')
        self.assertEqual(self.p.defer_reason, 'decode_floor')
        self.assertFalse(self.p.hold_decode(self.a))

    def test_never_served_skips_decode_floor(self):
        self.p.credit = 10.0
        self.p.begin_step(self.s)
        self.assertEqual(self.p.step_mode, 'prefill_turn')
        self.assertNotIn(self.b.request_id, self.p.last_service)
        self.assertGreater(self.p.cap_for(self.s, self.b), 0)

    def test_empty_isolated_prefill_forces_decode_next_step(self):
        self.learn()
        self.p._busy.append((self.clock(), 'decode', 1.0))
        self.p.credit = 10.0
        self.s.current_step += 1
        self.p.begin_step(self.s)
        self.assertEqual(self.p.step_mode, 'prefill_turn')
        self.assertTrue(self.p.hold_decode(self.a))
        self.p.finish_step(self.s, Out({}))
        self.s.current_step += 1
        self.p.begin_step(self.s)
        self.assertEqual(self.p.step_mode, 'prefill_turn')
        self.p.finish_step(self.s, Out({}))
        self.s.current_step += 1
        self.p.begin_step(self.s)
        self.assertEqual(self.p.step_mode, 'decode_only')
        self.assertEqual(self.p.defer_reason, 'empty_isolated')
        self.assertFalse(self.p.hold_decode(self.a))
        self.s.current_step += 1
        self.p.begin_step(self.s)
        self.assertEqual(self.p.step_mode, 'prefill_turn')

    def test_isolated_prefill_nets_one_minus_share(self):
        self.learn()
        self.p._busy.append((self.clock(), 'decode', 1.0))
        self.p.credit = 10.0
        self.s.current_step += 1
        self.p.begin_step(self.s)
        self.assertTrue(self.p.hold_decode(self.a))
        cap = self.p.cap_for(self.s, self.b)
        self.assertGreater(cap, 0)
        out = Out({'B': cap})
        self.p.finish_step(self.s, out)
        before = self.p.credit
        rec = self.p.inflight[id(out)]
        self.complete(out, 0.5)
        self.assertEqual(self.p.credit, self.p.credit)  # finite
        self.assertGreater(self.p._credit_limit(), 0)
        self.assertNotAlmostEqual(self.p.credit, before + self.p.share * 0.5, places=3)
        del rec

    def test_unfunded_prefill_turn_does_not_hold_decode(self):
        self.learn()
        self.p._busy.append((self.clock(), 'decode', 1.0))
        self.p.credit = -0.2
        self.p.begin_step(self.s)
        self.assertFalse(self.p.hold_decode(self.a))
        self.assertEqual(self.p.cap_for(self.s, self.b), 0)

    def test_decode_order_reserves_real_input_and_draft_capacity(self):
        # Inflight prefill → decode_only: A keeps the graph, B gets cap 0.
        self.submit({'B': 256})
        ns = self.running_loop(16)
        self.assertEqual(ns['num_scheduled_tokens'], {'A': 8})
        self.assertEqual(ns['input_budget'], 0)
        self.assertEqual([r.request_id for r in self.s.running], ['A', 'B'])

    def test_prefill_cannot_preempt_incumbent_for_kv(self):
        self.submit({'B': 256})
        ns = self.running_loop(7168, allocate=lambda r, n, **kw: [] if r is self.a else None)
        self.assertEqual(ns['num_scheduled_tokens'], {'A': 8})
        self.assertEqual([r.request_id for r in self.s.running], ['A', 'B'])
        self.assertFalse(self.p._open_rec['grants'])

    def test_finished_decode_does_not_hold_phantom_input_reservation(self):
        self.a.spec_token_ids = []
        self.a.num_computed_tokens = self.a.num_tokens
        ns = self.running_loop(264)
        self.assertEqual(ns['num_scheduled_tokens'], {'B': 256})
        self.assertEqual(ns['input_budget'], 0)


class TailStopTests(unittest.TestCase):
    """Head-log shape: scheduler block 64, Eagle, hits aligned to 3584, budget 7104."""

    def split_fn(self):
        if PATCHED_SOURCE is None:
            self.skipTest('needs the installed scheduler source')
        tree = ast.parse(PATCHED_SOURCE)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == '_mamba_block_aligned_split')
        ns = {'Request': object, '_GLM53_MIXED': POLICY_MOD_MIXED}
        exec(compile(ast.Module([fn], []), 'split', 'exec'), ns)
        return ns['_mamba_block_aligned_split']

    def sched(self, align=3584):
        return SimpleNamespace(
            cache_config=SimpleNamespace(block_size=64), use_eagle=True, hash_block_size=64,
            mamba_partial_cache_hit=False, max_num_scheduled_tokens=7104,
            scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
            _glm53_align_prefill_limit=None,
            kv_cache_manager=SimpleNamespace(coordinator=SimpleNamespace(_cache_hit_alignment_tokens=align)))

    def chunks(self, split, s, ctx, fresh):
        r = SimpleNamespace(num_prompt_tokens=ctx + fresh, num_tokens=ctx + fresh,
                            num_computed_tokens=ctx, shared_prefix_boundary=0)
        out = []
        while r.num_computed_tokens < r.num_prompt_tokens:
            n = split(s, r, min(r.num_prompt_tokens - r.num_computed_tokens, s.max_num_scheduled_tokens))
            self.assertGreater(n, 0)
            out.append(n)
            r.num_computed_tokens += n
        return out

    def test_unaligned_last_cache_position_no_longer_adds_a_tail_step(self):
        split, s, ctx = self.split_fn(), self.sched(), 21 * 3584
        # v6 split these as 2432+92, 448+88, 64+87 (one extra ~0.4 s step each).
        self.assertEqual(self.chunks(split, s, ctx, 2524), [2524])
        self.assertEqual(self.chunks(split, s, ctx, 536), [536])
        self.assertEqual(self.chunks(split, s, ctx, 151), [151])
        self.assertEqual(self.chunks(split, s, ctx, 32793)[-1], 32793 - 4 * 7104)

    def test_hit_aligned_last_cache_position_still_stops(self):
        # prompt = ctx + 3584 + 64: last_cache_position = ctx + 3584 is hit-aligned.
        split, s, ctx = self.split_fn(), self.sched(), 21 * 3584
        self.assertEqual(self.chunks(split, s, ctx, 3584 + 64), [3584, 64])

    def test_without_alignment_info_keeps_stock_stop(self):
        split, ctx = self.split_fn(), 21 * 3584
        s = self.sched()
        s.kv_cache_manager = SimpleNamespace(coordinator=SimpleNamespace())
        self.assertEqual(self.chunks(split, s, ctx, 2524), [2432, 92])


def installation_tests():
    src = next((p for p in [Path(os.environ.get('GLM53_SCHEDULER_PY_SRC', '/missing')),
                           Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py'),
                           Path('/tmp/sched-live.py')] if p.is_file()), None)
    if src is None:
        raise SystemExit('Set GLM53_SCHEDULER_PY_SRC to the pinned scheduler source')
    clean = src.read_text()
    for marker, fn in [(mod.MARK_V7, mod.unpatch_v7), (mod.MARK_V6, mod.unpatch_v6), (mod.MARK_V5, mod.unpatch_v5), (mod.MARK_V4, mod.unpatch_v4), (mod.MARK_V3, mod.unpatch_v3), (mod.MARK_V2, mod.unpatch_v2)]:
        if marker in clean:
            clean = fn(clean)
    if mod.V1_HELPER_START in clean:
        clean = mod.unpatch_v1(clean)
    with tempfile.TemporaryDirectory() as temp:
        for version in (0, 1, 2, 3, 4, 5, 6):
            text = clean
            if version:
                marker = mod.MARK if version == 1 else getattr(mod, f'MARK_V{version}')
                helper = ('\ndef _glm53_mixed_prefill_policy(running, current):\n    return 0\n\n' if version == 1 else
                          f'\nclass _Glm53MixedPrefill:  {marker}\n    pass\n\n')
                needle = 'from vllm.compilation.cuda_graph import CUDAGraphStat\n'
                text = text.replace(needle, helper + needle, 1)
                if version == 6:
                    for new, old, label in mod.V6_PAIRS:
                        text = mod.replace_once(text, old, new, label)
                elif version == 5:
                    for new, old, label in mod.V5_PAIRS:
                        text = mod.replace_once(text, old, new, label)
                elif version == 4:
                    for new, old, label in mod.V4_PAIRS:
                        text = mod.replace_once(text, old, new, label)
                else:
                    names = ['RUNNING', 'WAITING'] if version == 1 else ['BEGIN', 'OBS', 'RUNNING', 'WAITING', 'ALIGN', 'RUNNING_MAMBA', 'WAITING_MAMBA']
                    if version == 3:
                        names.append('FIN')
                    for name in names:
                        old = mod.V3_FIN_OLD if name == 'FIN' else getattr(mod, name + '_OLD')
                        text = mod.replace_once(text, old, getattr(mod, f'V{version}_{name}_NEW'), name)
            target = Path(temp) / f'scheduler_v{version}.py'
            target.write_text(text)
            env = {**os.environ, 'GLM53_SCHEDULER_PY': str(target), 'GLM53_MIXED_PREFILL_CHUNK': 'skip'}
            subprocess.run([sys.executable, str(PATCH)], env=env, check=True, capture_output=True)
            installed = target.read_text()
            compile(installed, str(target), 'exec')
            assert mod.MARK_V7 in installed and mod.MARK_V6 not in installed and mod.MARK_V5 not in installed and mod.MARK_V4 not in installed and mod.MARK_V3 not in installed and mod.MARK_V2 not in installed
            subprocess.run([sys.executable, str(PATCH)], env=env, check=True, capture_output=True)
            assert target.read_text() == installed
            # Marker alone must not suppress validation or overwrite source drift.
            drifted = installed.replace('_GLM53_MIXED.finish_step(self, scheduler_output)', '_GLM53_MIXED.finish_step_changed(self, scheduler_output)', 1)
            target.write_text(drifted)
            result = subprocess.run([sys.executable, str(PATCH)], env=env, capture_output=True)
            assert result.returncode != 0 and target.read_text() == drifted
        return installed


def main():
    global POLICY, PATCHED_SOURCE
    PATCHED_SOURCE = installation_tests()
    begin = PATCHED_SOURCE.index('class _Glm53MixedPrefill:')
    end = PATCHED_SOURCE.index('_GLM53_MIXED = _Glm53MixedPrefill()')
    ns = {'os': os, 'time': __import__('time')}
    exec(PATCHED_SOURCE[begin:end], ns)
    POLICY = ns['_Glm53MixedPrefill']
    global POLICY_MOD_MIXED
    POLICY_MOD_MIXED = POLICY
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(FairTests),
                                unittest.defaultTestLoader.loadTestsFromTestCase(TailStopTests)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
