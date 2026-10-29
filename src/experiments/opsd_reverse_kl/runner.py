"""Frozen D4 to one sampled intermediate depth per update, plus frozen D4 anchor."""

import argparse, contextlib, datetime as dt, hashlib, json, math, os, random
import signal, sys, time
from pathlib import Path
import torch
from phase_timing import PhaseTimes

TIMES = PhaseTimes()
from experiments.common.rollouts import pack_rollouts, forward_kl
from experiments.opsd_reverse_kl.loss import backward_opsd, backward_reference
from experiments.common.native import (
    NativeForward,
    collect_sampled_rollout,
)
from experiments.common.gradient_diagnostics import ComponentSnapshot
from experiments.opsd_sampled_preservation.schedule import shallow_jobs_for_update
import recovery
from native_recurrent import NativeReadout, prepare_data
from train import seed_all, batch_tensors, eval_nll
from native_recipe_eval import eval_generation


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def all_digest(model):
    h = hashlib.sha256()
    for n, p in model.base.state_dict().items():
        h.update(n.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_native(model_dir, cfg, checkpoint, expected_sha, device):
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    )
    if checkpoint:
        assert (
            expected_sha and digest(checkpoint) == expected_sha
        ), "SFT checkpoint hash mismatch"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert (
            payload["step"] == 64
        ), "Use successful final SFT, never a profile checkpoint"
        base.load_state_dict(payload["model"], strict=True)
        del payload
    model = NativeReadout(base, cfg["native_kind"], "core", "final-depth-ce")
    model.checkpoint_layers = True
    model.evaluation_protocol = "native-depth-metrics-v2"
    return model.to(device)


def make_teacher(args, cfg, student):
    assert cfg["teacher_mode"] in ("frozen", "sft-only"), "Shared teachers retired"
    if cfg["teacher_mode"] == "sft-only":
        return student
    return (
        load_native(
            args.model_dir, cfg, args.teacher_checkpoint, args.teacher_sha, "cpu"
        )
        .requires_grad_(False)
        .eval()
    )


def teacher_target(teacher, rollout, context, depth, amp):
    frozen = not any(p.requires_grad for p in teacher.parameters())
    if frozen:
        with TIMES.measure("teacher_transfer_seconds"):
            teacher.to("cuda")
    fwd = NativeForward(teacher)
    try:
        with TIMES.measure("teacher_forward_seconds"), torch.no_grad(), fwd.replay_mode(
            rollout.inputs, context
        ), amp():
            target = fwd(rollout.inputs, [depth], rollout.prediction_mask, context)[
                depth
            ].detach()
        with TIMES.measure("target_transfer_seconds"):
            target = target.cpu()
    finally:
        if frozen:
            with TIMES.measure("teacher_transfer_seconds"):
                teacher.to("cpu")
            torch.cuda.empty_cache()
    assert not target.requires_grad
    if frozen:
        assert not any(p.grad is not None for p in teacher.parameters())
    return target


def gate(args, cfg, output):
    from transformers import AutoTokenizer
    from base_depth_eval import states
    import bitsandbytes as bnb

    seed_all(cfg["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    )
    enc, rows, eos = prepare_data(args.data_dir, tokenizer, cfg, output)
    student = load_native(
        args.model_dir, cfg, args.student_checkpoint, args.student_sha, "cuda"
    )
    teacher = make_teacher(args, cfg, student)
    frozen = student.frozen_digest()
    before = all_digest(teacher)
    amp = lambda: torch.autocast("cuda", dtype=torch.float16, cache_enabled=False)
    fwd = NativeForward(student)
    longest = max(
        rows["train"],
        key=lambda r: len(tokenizer.encode(r["prompt"], add_special_tokens=False)),
    )
    parity = []
    stress_batches = {}
    depths = cfg["student_depths"] or [4]
    for depth in sorted(set(depths + ([4] if cfg["preservation_weight"] else []))):
        short = collect_sampled_rollout(
            student, longest, tokenizer, depth, eos, limit=8, trace=True
        )
        batch = pack_rollouts(
            [short.prefix], [short.completion], eos=eos, pad=eos, device="cuda"
        )
        with torch.no_grad(), fwd.replay_mode(batch.inputs, short.initial_state), amp():
            replay = fwd(
                batch.inputs, [depth], batch.prediction_mask, short.initial_state
            )[depth].cpu()
        error = float((replay - short.cached_logits).abs().max())
        kl = float(forward_kl(replay, short.cached_logits).abs().max())
        torch.testing.assert_close(replay, short.cached_logits, rtol=0.005, atol=0.02)
        assert kl < 1e-4
        equal = teacher_target(teacher, batch, short.initial_state, depth, amp)
        assert torch.equal(replay, equal), "Same initial checkpoint/depth parity"
        parity.append({"depth": depth, "max_abs": error, "max_forward_kl": kl})
        completion = short.completion[:-1] if short.ended else short.completion
        completion = completion or [int(replay[0].argmax())]
        completion = [
            v if v != eos else (eos + 1) % replay.shape[-1] for v in completion
        ]
        completion = (completion * math.ceil(512 / len(completion)))[:512]
        stress = pack_rollouts(
            [short.prefix], [completion], eos=eos, pad=eos, device="cuda"
        )
        state = states(
            student.base, cfg["native_kind"], longest, len(short.prefix) + 512
        )
        stress_batches[depth] = (
            stress,
            None if state is None else state.detach().cpu(),
        )
    opt = bnb.optim.AdamW8bit(
        [p for p in student.parameters() if p.requires_grad],
        lr=cfg["learning_rate"],
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=128)
    torch.cuda.reset_peak_memory_stats()
    updates = 0
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        for depth in depths:
            stress, state = stress_batches[depth]
            target = teacher_target(teacher, stress, state, 4, amp)
            backward_opsd(
                fwd,
                target,
                stress,
                depth=depth,
                denominator=len(depths),
                backward=lambda l: scaler.scale(l).backward(),
                replay_context=state,
                autocast=amp,
            )
        probe = ComponentSnapshot(
            [p for p in student.parameters() if p.requires_grad],
            output / "gate-gradient-probe",
        )
        probe.capture()
        stress, state = stress_batches[4]
        target = teacher_target(teacher, stress, state, 4, amp)
        backward_opsd(
            fwd,
            target,
            stress,
            depth=4,
            denominator=1,
            backward=lambda l: scaler.scale(l).backward(),
            replay_context=state,
            autocast=amp,
        )
        gate_diagnostics = probe.combine(scaler.get_scale())
        probe.close()
        scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        assert torch.isfinite(norm) and norm > 0
        scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        assert scaler.get_scale() >= scale
        updates += 1
    assert student.frozen_digest() == frozen
    if cfg["teacher_mode"] == "frozen":
        assert all_digest(teacher) == before
    else:
        assert teacher is student and all_digest(teacher) != before
    peak = torch.cuda.max_memory_allocated() / 2**30
    assert peak < torch.cuda.get_device_properties(0).total_memory / 2**30 - 0.75
    result = {
        "status": "passed",
        "parity_by_depth": parity,
        "teacher_mode": cfg["teacher_mode"],
        "teacher_fixed": cfg["teacher_mode"] == "frozen",
        "shared_parameter_identity_checked": teacher is student,
        "student_frozen_scope_unchanged": True,
        "stress_response_tokens": 512,
        "adamw8bit_updates": updates,
        "peak_gpu_gib": peak,
        "gradient_probe_memory_stress_checked": bool(cfg["student_depths"]),
        "profile_weights_discarded": True,
        "config_sha256": digest(args.config),
        "teacher_checkpoint_sha256": args.teacher_sha,
        "student_checkpoint_sha256": args.student_sha,
    }
    (output / "memory-gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def run(args, cfg, out):
    from transformers import AutoTokenizer

    assert (
        cfg["seed"] == 42
        and cfg["maximum_depth"] == 4
        and set(cfg["student_depths"]) <= {1, 2, 3}
    )
    assert cfg["teacher_mode"] in ("frozen", "sft-only")
    assert cfg["student_depths"] == [1, 2, 3] and cfg["teacher_mode"] == "frozen"
    assert cfg["student_depth_sampling"] == "uniform-once-per-optimizer-update"
    assert cfg["shallow_loss_weights"] == {"1": 1 / 3, "2": 1 / 3, "3": 1 / 3}
    assert cfg["preservation_weight"] == 1 and cfg["reference_ce_weight"] == 0
    assert cfg["opsd_weight"] == 1.0
    assert not cfg["teacher_privileged_context"]
    assert cfg["optimizer"] == "adamw8" and cfg["eval_depths"] == [1, 2, 4]
    gate_record = json.load(open(args.gate))
    assert gate_record["status"] == "passed" and gate_record["config_sha256"] == digest(
        args.config
    )
    assert (
        gate_record["teacher_checkpoint_sha256"] == args.teacher_sha
        and gate_record["student_checkpoint_sha256"] == args.student_sha
    )
    seed_all(cfg["seed"])
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    )
    enc, rows, eos = prepare_data(args.data_dir, tokenizer, cfg, out)
    student = load_native(
        args.model_dir, cfg, args.student_checkpoint, args.student_sha, "cuda"
    )
    teacher = make_teacher(args, cfg, student)
    initial_teacher_hash = all_digest(teacher)
    frozen = student.frozen_digest()
    amp = lambda: torch.autocast("cuda", dtype=torch.float16, cache_enabled=False)
    fwd = NativeForward(student)
    import bitsandbytes as bnb

    opt = bnb.optim.AdamW8bit(
        [p for p in student.parameters() if p.requires_grad],
        lr=cfg["learning_rate"],
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=128)
    rng = random.Random(cfg["seed"])
    cursor = recovery.DataCursor(len(rows["train"]), 16, 1, cfg["seed"] + 200)
    identity = hashlib.sha256(
        json.dumps(
            {
                "cfg": cfg,
                "teacher": args.teacher_sha,
                "student": args.student_sha,
                "sources": {
                    str(p.relative_to(Path(__file__).parents[1])): digest(p)
                    for folder in [Path(__file__).parent, Path(__file__).parents[1]/"common"]
                    for p in folder.glob("*.py")
                },
                "native": {
                    n: digest(Path(__import__(n).__file__))
                    for n in [
                        "train",
                        "native_recurrent",
                        "native_recipe_eval",
                        "base_depth_eval",
                        "recovery",
                    ]
                },
                "rollout_loss": digest(Path(__file__).parents[1] / "common/rollouts.py"),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    progress = {
        "step": 0,
        "evaluation_steps": [],
        "metrics_offset": 0,
        "prediction_offset": 0,
        "rollouts_offset": 0,
        "skipped_updates": 0,
    }
    if args.resume:
        state, progress = recovery.load(
            args.resume, student, opt, scaler, rng, identity
        )
        cursor = recovery.DataCursor(
            len(rows["train"]), 16, 1, cfg["seed"] + 200, state
        )
        for file, key in [
            ("metrics.jsonl", "metrics_offset"),
            ("predictions.jsonl", "prediction_offset"),
            ("rollouts.jsonl", "rollouts_offset"),
        ]:
            path = out / file
            if path.exists():
                assert path.stat().st_size >= progress[key]
                with path.open("r+b") as stream:
                    stream.truncate(progress[key])
    elif (out / "metrics.jsonl").exists():
        raise RuntimeError("Existing run: resume, never append a fresh run")
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "teacher_checkpoint_sha256": args.teacher_sha,
                "student_checkpoint_sha256": args.student_sha,
                "teacher_fixed": cfg["teacher_mode"] == "frozen",
                "teacher_mode": cfg["teacher_mode"],
                "teacher_initial_weights_sha256": initial_teacher_hash,
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "seed": cfg["seed"],
                "scope": "core",
                "teacher_cpu_offload_between_forwards": cfg["teacher_mode"] == "frozen",
                "identity": identity,
            },
            indent=2,
        )
        + "\n"
    )
    scratch = out / "gradient-probe-scratch"
    if scratch.exists():
        import shutil

        shutil.rmtree(scratch)
    started = time.monotonic()
    deadline = started + cfg["max_runtime_seconds"]
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda *a: stopped.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *a: stopped.__setitem__(0, True))
    log = (out / "metrics.jsonl").open("a", buffering=1)
    rollog = (out / "rollouts.jsonl").open("a", buffering=1)
    tracker = None
    if os.environ.get("WANDB_API_KEY"):
        import wandb

        tracker = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "looped-opsd"),
            entity=os.environ.get("WANDB_ENTITY"),
            group="GSM8K/OPSD-sampled123-reverseKL",
            name=cfg["run_id"],
            config=cfg,
            id=progress.get("wandb_id"),
            resume="allow",
        )
        tracker.define_metric("step")
        tracker.define_metric("*", step_metric="step")
        progress["wandb_id"] = tracker.id
        (out / "tracking.json").write_text(json.dumps({"run_id": tracker.id}) + "\n")

    def emit(event):
        event["wall_seconds"] = time.monotonic() - started
        log.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)
        if tracker:
            flat = {k: v for k, v in event.items() if isinstance(v, (int, float))}
            for d, metrics in event.get("metrics", {}).items():
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        flat[event["event"] + "/depth_" + str(d) + "/" + k] = v
            tracker.log(flat)

    def save():
        log.flush()
        rollog.flush()
        progress["metrics_offset"] = (out / "metrics.jsonl").stat().st_size
        progress["prediction_offset"] = (
            (out / "predictions.jsonl").stat().st_size
            if (out / "predictions.jsonl").exists()
            else 0
        )
        progress["rollouts_offset"] = (out / "rollouts.jsonl").stat().st_size
        recovery.save(
            out / "recovery.pt", student, opt, scaler, cursor, rng, progress, identity
        )

    def evaluate():
        step = progress["step"]
        if step in progress["evaluation_steps"]:
            return
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        emit(
            {
                "event": "validation",
                "step": step,
                "metrics": eval_nll(
                    student, enc["validation"], [1, 2, 4], torch.float16, 2468, eos
                ),
            }
        )
        for family, data in [
            ("validation_generation", rows["validation"]),
            ("train_generation", rows["train"][:16]),
        ]:
            metrics = eval_generation(
                student,
                data,
                tokenizer,
                [1, 2, 4],
                torch.float16,
                2468,
                eos,
                512,
                out,
                step,
                family,
                deadline,
                batch_size=cfg["generation_batch_size"],
                should_stop=lambda: stopped[0],
            )
            emit({"event": family, "step": step, "metrics": metrics})
            if family == "validation_generation":
                progress["final_metrics"] = metrics
        if step == 0:
            assert args.start_predictions, 'Supply common-SFT predictions.jsonl for matched-start verification'
            assert args.student_sha == args.teacher_sha, 'Student and teacher must use the common SFT'
            source = Path(args.start_predictions)
            records = [json.loads(line) for line in source.read_text().splitlines()]
            records = [r for r in records if r['step']==64 and r['family']=='final_validation_generation']
            reference = {'records': records, 'source_sha256': digest(source)}
            expected = {
                (r["depth"], r["question_sha256"]): r for r in reference["records"]
            }
            actual = [
                json.loads(x)
                for x in (out / "predictions.jsonl").read_text().splitlines()
            ]
            actual = [
                r
                for r in actual
                if r["step"] == 0 and r["family"] == "validation_generation"
            ]
            assert len(actual) == 768 and len(expected) == 768
            differences = []
            for r in actual:
                ref = expected[(r["depth"], r["question_sha256"])]
                for k in [
                    "correct",
                    "numerically_correct",
                    "eos",
                    "truncated",
                    "generated_tokens",
                ]:
                    if r[k] != ref[k]:
                        differences.append(
                            {
                                "depth": r["depth"],
                                "question_sha256": r["question_sha256"],
                                "field": k,
                                "old": ref[k],
                                "new": r[k],
                            }
                        )
            (out / "matched-start-audit.json").write_text(
                json.dumps(
                    {
                        "matched": not differences,
                        "differences": differences,
                        "reference": reference["source_sha256"],
                    },
                    indent=2,
                )
                + "\n"
            )
            assert (
                not differences
            ), "Same-start evaluation differs from matched SD baseline; hold before update1"
        progress["evaluation_steps"].append(step)
        save()

    try:
        if not args.resume:
            save()
        while True:
            if progress["step"] in [0, 32, 64]:
                evaluate()
            indices = cursor.batch()
            if indices is None:
                break
            if stopped[0] or time.monotonic() > deadline:
                raise InterruptedError(
                    "Worker lease/runtime reached; recover from durable update boundary"
                )
            TIMES.reset_step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_started = time.perf_counter()
            step = progress["step"] + 1
            # All trajectories are drawn before an update; no optimizer change
            # or reference supervision occurs between sampling and replay.
            shallow_jobs = shallow_jobs_for_update(indices, rng, cfg["student_depths"])
            depths = [d for i, d in shallow_jobs]
            with TIMES.measure("rollout_seconds"):
                rollouts = [
                    collect_sampled_rollout(
                        student,
                        rows["train"][i],
                        tokenizer,
                        d,
                        eos,
                        deadline=deadline,
                        should_stop=lambda: stopped[0],
                    )
                    for i, d in shallow_jobs
                ]
                deep_rollouts = (
                    [
                        collect_sampled_rollout(
                            student,
                            rows["train"][i],
                            tokenizer,
                            4,
                            eos,
                            deadline=deadline,
                            should_stop=lambda: stopped[0],
                        )
                        for i in indices
                    ]
                    if cfg["preservation_weight"]
                    else []
                )
            opt.zero_grad(set_to_none=True)
            student.train()
            lr = cfg["learning_rate"] * (
                step / 8
                if step <= 8
                else 0.5 * (1 + math.cos(math.pi * (step - 8) / 56))
            )
            for g in opt.param_groups:
                g["lr"] = lr
            totals = {
                "kl_sequence_mean_sum": 0.0,
                "kl_sum": 0.0,
                "rollout_tokens": 0,
                "rollout_examples": 0,
                "reference_ce_sum": 0.0,
                "reference_tokens": 0,
            }
            refden = sum(sum(y != -100 for y in enc["train"][i][1]) for i in indices)
            preservation_sum = 0.0
            preservation_tokens = 0
            depth_kl_sums = {1: 0.0, 2: 0.0, 3: 0.0}
            probe = bool(
                rollouts
                and (deep_rollouts or cfg["reference_ce_weight"])
                and (step == 1 or step % cfg["gradient_cosine_every"] == 0)
            )
            snapshot = (
                ComponentSnapshot(
                    [p for p in student.parameters() if p.requires_grad], scratch
                )
                if probe
                else None
            )
            diagnostics = {"gradient_probe_measured": probe}
            try:
                # Full-batch shallow component first. All rollouts were sampled
                # before either loss; weights remain unchanged until one step.
                for j, (i, depth) in enumerate(shallow_jobs):
                    r = rollouts[j]
                    packed = pack_rollouts(
                        [r.prefix], [r.completion], eos=eos, pad=eos, device="cuda"
                    )
                    target = teacher_target(teacher, packed, r.initial_state, 4, amp)
                    stats = backward_opsd(
                        fwd,
                        target,
                        packed,
                        depth=depth,
                        denominator=len(shallow_jobs),
                        backward=lambda l: scaler.scale(l).backward(),
                        replay_context=r.initial_state,
                        autocast=amp,
                    )
                    for k, v in stats.items():
                        totals[k] = totals.get(k, 0) + v
                    depth_kl_sums[depth] += stats["kl_sequence_mean_sum"]
                    rollog.write(
                        json.dumps(
                            {
                                "step": step,
                                "question_sha256": rows["train"][i]["question_sha256"],
                                "completion_tokens": r.completion,
                                "ended": r.ended,
                                "student_depth": depth,
                                "family": "shallow_opsd",
                            }
                        )
                        + "\n"
                    )
                    del target, packed
                if snapshot:
                    snapshot.capture()
                for j, i in enumerate(indices):
                    if deep_rollouts:
                        r = deep_rollouts[j]
                        packed = pack_rollouts(
                            [r.prefix], [r.completion], eos=eos, pad=eos, device="cuda"
                        )
                        target = teacher_target(
                            teacher, packed, r.initial_state, 4, amp
                        )
                        stats = backward_opsd(
                            fwd,
                            target,
                            packed,
                            depth=4,
                            denominator=len(indices),
                            backward=lambda l: scaler.scale(
                                cfg["preservation_weight"] * l
                            ).backward(),
                            replay_context=r.initial_state,
                            autocast=amp,
                        )
                        for timing_key in [
                            "student_forward_seconds",
                            "backward_seconds",
                        ]:
                            totals[timing_key] = (
                                totals.get(timing_key, 0) + stats[timing_key]
                            )
                        preservation_sum += stats["kl_sequence_mean_sum"]
                        preservation_tokens += stats["rollout_tokens"]
                        rollog.write(
                            json.dumps(
                                {
                                    "step": step,
                                    "question_sha256": rows["train"][i][
                                        "question_sha256"
                                    ],
                                    "completion_tokens": r.completion,
                                    "ended": r.ended,
                                    "student_depth": 4,
                                    "family": "preservation",
                                }
                            )
                            + "\n"
                        )
                        del target, packed
                    if cfg["reference_ce_weight"]:
                        x, y = batch_tensors([enc["train"][i]], "cuda", eos)
                        stats = backward_reference(
                            fwd,
                            x,
                            y,
                            depth=4,
                            denominator=refden,
                            weight=cfg["reference_ce_weight"],
                            backward=lambda l: scaler.scale(l).backward(),
                            autocast=amp,
                        )
                        for k, v in stats.items():
                            totals[k] = totals.get(k, 0) + v
                if snapshot:
                    diagnostics.update(snapshot.combine(scaler.get_scale()))
            finally:
                if snapshot:
                    snapshot.close()
            optimizer_started = time.perf_counter()
            scaler.unscale_(opt)
            norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            assert torch.isfinite(norm), "Nonfinite gradient"
            oldscale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() < oldscale:
                raise FloatingPointError(
                    "Skipped update: hold for inspection, do not silently change exposure"
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            TIMES.step["optimizer_seconds"] = time.perf_counter() - optimizer_started
            progress["step"] = step
            cursor.advance(len(indices))
            emit(
                {
                    "event": "train",
                    "train_step_seconds": time.perf_counter() - step_started,
                    **TIMES.step,
                    "step": step,
                    "opsd_loss": totals["kl_sequence_mean_sum"] / len(shallow_jobs),
                    "reference_ce": totals["reference_ce_sum"]
                    / max(1, totals["reference_tokens"]),
                    "gradient_norm": float(norm),
                    "gradient_clip_multiplier": min(1.0, 1.0 / max(float(norm), 1e-30)),
                    "lr": lr,
                    "eos_rate": sum(r.ended for r in rollouts) / max(1, len(rollouts)),
                    "mean_rollout_length": totals["rollout_tokens"] / len(shallow_jobs),
                    "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "shallow_kl": totals["kl_sequence_mean_sum"] / len(shallow_jobs),
                    "preservation_kl": preservation_sum / len(indices),
                    "weighted_preservation_kl": cfg["preservation_weight"]
                    * preservation_sum
                    / len(indices),
                    "total_objective": totals["kl_sequence_mean_sum"]
                    / len(shallow_jobs)
                    + cfg["preservation_weight"] * preservation_sum / len(indices)
                    + cfg["reference_ce_weight"]
                    * totals["reference_ce_sum"]
                    / max(1, totals["reference_tokens"]),
                    "preservation_eos_rate": (
                        sum(r.ended for r in deep_rollouts) / len(deep_rollouts)
                        if deep_rollouts
                        else None
                    ),
                    "preservation_mean_rollout_length": (
                        preservation_tokens / len(deep_rollouts)
                        if deep_rollouts
                        else None
                    ),
                    "shallow_kl_d1": (
                        depth_kl_sums[1] / len(indices) if depths[0] == 1 else None
                    ),
                    "shallow_kl_d2": (
                        depth_kl_sums[2] / len(indices) if depths[0] == 2 else None
                    ),
                    "shallow_kl_d3": (
                        depth_kl_sums[3] / len(indices) if depths[0] == 3 else None
                    ),
                    "sampled_student_depth": depths[0],
                    "anchor_kind": "forward_KL4",
                    "shallow_kl_direction": "student_to_teacher",
                    **diagnostics,
                    "preservation_tokens": preservation_tokens,
                    "depth_counts": {str(d): depths.count(d) for d in [1, 2, 3]},
                    **totals,
                }
            )
            del rollouts, deep_rollouts
            opt.zero_grad(set_to_none=True)
            if step % 8 == 0:
                save()
            if step == 64:
                recovery.atomic_save(
                    {"model": student.base.state_dict(), "step": step},
                    out / ("checkpoint-step" + str(step) + ".pt"),
                )
        assert progress["step"] == 64 and student.frozen_digest() == frozen
        if cfg["teacher_mode"] == "frozen":
            assert all_digest(teacher) == initial_teacher_hash
        (out / "frozen-weight-audit.json").write_text(
            json.dumps(
                {
                    "teacher_unchanged": all_digest(teacher) == initial_teacher_hash,
                    "teacher_mode": cfg["teacher_mode"],
                    "shared_parameter_identity": teacher is student,
                    "student_frozen_scope_unchanged": True,
                }
            )
            + "\n"
        )
        (out / "summary.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "steps": 64,
                    "skipped_updates": 0,
                    "final_generation": progress["final_metrics"],
                    "teacher_fixed": cfg["teacher_mode"] == "frozen",
                    "teacher_mode": cfg["teacher_mode"],
                    "teacher_checkpoint_sha256": args.teacher_sha,
                    "student_checkpoint_sha256": args.student_sha,
                },
                indent=2,
            )
            + "\n"
        )
    except BaseException as exc:
        # Do not save a partial update or mutated/overflow optimizer state.
        (out / "failure.json").write_text(
            json.dumps(
                {"error": str(exc), "last_durable_checkpoint": str(out / "recovery.pt")}
            )
            + "\n"
        )
        raise
    finally:
        log.close()
        rollog.close()
        if tracker:
            tracker.finish()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output", required=True)
    for role in ["student", "teacher"]:
        p.add_argument("--" + role + "-checkpoint")
        p.add_argument("--" + role + "-sha")
    p.add_argument("--start-predictions", help="Common-SFT predictions.jsonl, including final step-64 validation")
    p.add_argument("--gate")
    p.add_argument("--gate-only", action="store_true")
    p.add_argument("--resume")
    a = p.parse_args()
    cfg = json.load(open(a.config))
    out = Path(a.output)
    with recovery.exclusive_run(out):
        if a.gate_only:
            gate(a, cfg, out)
        else:
            assert a.gate, "Native CUDA gate required"
            run(a, cfg, out)


if __name__ == "__main__":
    main()
