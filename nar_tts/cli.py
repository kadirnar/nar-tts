"""Command-line interface for inference and quality tooling."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path


def _json_value(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid JSON: {error}") from error


def _control_from_args(args):
    from nar_tts.core.controls import SpeechControl, VocalEvent

    events = []
    for value in args.event or ():
        if not isinstance(value, dict):
            raise TypeError("--event must be a JSON object")
        events.append(VocalEvent.from_dict(value))
    return SpeechControl(
        emotion=args.emotion,
        intensity=args.intensity,
        delivery=args.delivery,
        valence=args.valence,
        arousal=args.arousal,
        events=tuple(events),
    )


def _infer(args):
    from nar_tts.evaluation.data_quality import read_jsonl
    from nar_tts.inference.infer import (
        NarTTS,
        SynthesisRequest,
        load_inference_config,
    )

    if args.manifest:
        source = Path(args.manifest).resolve()
        rows = list(read_jsonl(source))
        requests = []
        for index, row in enumerate(rows):
            value = dict(row)
            value.setdefault("id", str(index))
            if "control" not in value:
                control = {
                    name: value.pop(name)
                    for name in (
                        "emotion",
                        "intensity",
                        "delivery",
                        "valence",
                        "arousal",
                        "events",
                    )
                    if name in value
                }
                if control:
                    value["control"] = control
            if "ref_wav" in value and "reference_audio" not in value:
                value["reference_audio"] = value.pop("ref_wav")
            if "ref_text" in value and "reference_text" not in value:
                value["reference_text"] = value.pop("ref_text")
            if args.output_dir:
                value["output_path"] = os.fspath(
                    Path(args.output_dir) / f"{value['id']}.wav"
                )
            elif (
                value.get("output_path")
                and not Path(value["output_path"]).is_absolute()
            ):
                value["output_path"] = os.fspath(source.parent / value["output_path"])
            if not value.get("output_path"):
                raise ValueError(
                    "batch rows need output_path or the command needs --output-dir"
                )
            reference = value.get("reference_audio")
            if reference and not Path(reference).is_absolute():
                value["reference_audio"] = os.fspath(source.parent / reference)
            requests.append(SynthesisRequest(**value))
    else:
        missing = [
            name
            for name, value in (
                ("--text", args.text),
                ("--reference", args.reference),
                ("--reference-text", args.reference_text),
                ("--output", args.output),
            )
            if not value
        ]
        if missing:
            raise ValueError("direct inference requires " + ", ".join(missing))
        requests = [
            SynthesisRequest(
                text=args.text,
                reference_audio=args.reference,
                reference_text=args.reference_text,
                output_path=args.output,
                id=args.id,
                language=args.language,
                control=_control_from_args(args),
                target_duration_seconds=args.target_duration,
                seed=args.seed,
            )
        ]

    if args.long_form and len(requests) != 1:
        raise ValueError("--long-form currently accepts one direct request")
    settings = load_inference_config(args.config)
    if args.text_eos_token_id is not None:
        settings["tokens"]["text_eos_token_id"] = args.text_eos_token_id
    if args.pad_token_id is not None:
        settings["tokens"]["pad_token_id"] = args.pad_token_id
    engine = NarTTS(
        checkpoint=args.checkpoint,
        tokenizer_name=args.tokenizer,
        device=args.device,
        config=settings,
    )
    if args.long_form:

        def progress(index, result):
            seconds = len(result.waveform) / result.sample_rate
            print(f"chunk {index + 1}: {seconds:.2f}s", flush=True)

        results = [engine.synthesize_long(requests[0], on_chunk=progress)]
    else:
        results = engine.synthesize_batch(requests)
    print(
        json.dumps(
            [
                {
                    "id": result.request.id,
                    "output": result.winner.audio_path if result.winner else None,
                    "accepted": result.accepted,
                    "score": result.winner.score if result.winner else None,
                    "report": result.report_path,
                    "real_time_factor": result.timings.get("real_time_factor"),
                }
                for result in results
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    if any(not result.accepted for result in results):
        return 2
    return 0


def _evaluate(args):
    from nar_tts.evaluation.reporting import evaluate_manifest

    asr_config = None
    if not args.no_asr:
        asr_config = {
            "model": args.asr_model,
            "device": args.device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "language": args.language,
        }
    report = evaluate_manifest(
        args.manifest,
        args.output,
        asr_config=asr_config,
        listening_manifest=args.listening_manifest,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


def _audit_data(args):
    from nar_tts.evaluation.data_quality import AuditThresholds, audit_manifest

    thresholds = AuditThresholds(
        minimum_seconds=args.minimum_seconds,
        maximum_seconds=args.maximum_seconds,
        maximum_clipping_ratio=args.maximum_clipping_ratio,
        maximum_silence_ratio=args.maximum_silence_ratio,
        require_license=args.require_license,
    )
    report = audit_manifest(
        args.manifest,
        args.accepted,
        args.rejected,
        thresholds=thresholds,
        text_column=args.text_column,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _distill(args):
    from nar_tts.evaluation.data_quality import build_distillation_manifest

    paths = []
    for pattern in args.reports:
        matches = sorted(glob.glob(pattern, recursive=True))
        paths.extend(matches or [pattern])
    report = build_distillation_manifest(
        paths,
        args.output,
        minimum_score=args.minimum_score,
        maximum_cer=args.maximum_cer,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _encode_expressive(args):
    from nar_tts.preprocessing.expressive import encode_expressive_manifest

    report = encode_expressive_manifest(
        args.manifest,
        args.output,
        tokenizer_name=args.tokenizer,
        checkpoint=args.checkpoint,
        device=args.device,
        codec_model=args.codec,
        codec_dtype=args.dtype,
        batch_size=args.batch_size,
        tag_neutral=args.tag_neutral,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _codec_check(args):
    from nar_tts.evaluation.codec import codec_reconstruction_report

    paths = []
    for pattern in args.audio:
        matches = sorted(glob.glob(pattern, recursive=True))
        paths.extend(matches or [pattern])
    report = codec_reconstruction_report(
        paths,
        args.output,
        device=args.device,
        model=args.codec,
        dtype=args.dtype,
        num_codebooks=args.num_codebooks,
        batch_size=args.batch_size,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


def _preprocess(args):
    from nar_tts.preprocessing.encode_pretrain import main as preprocess_main

    preprocess_main(["--config", args.config])
    return 0


def _inspect_tokenizer(args):
    from transformers import AutoTokenizer

    from nar_tts.core.tokens import TokenLayout

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    layout = TokenLayout.from_tokenizer(tokenizer)
    print(
        json.dumps(
            {
                "model": args.model,
                "vocabulary_size": len(tokenizer),
                "text_eos_token_id": layout.eot,
                "pad_token_id": tokenizer.pad_token_id,
                "suggested_pad_token_id": (
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else layout.eot
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _parser():
    parser = argparse.ArgumentParser(
        prog="nar-tts", description="Nar TTS quality tools"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer", help="voice cloning with adaptive Best-of-N")
    infer.add_argument("--config", default=None)
    infer.add_argument("--checkpoint")
    infer.add_argument("--tokenizer")
    infer.add_argument("--device")
    infer.add_argument("--text-eos-token-id", type=int)
    infer.add_argument("--pad-token-id", type=int)
    infer.add_argument("--manifest", help="batch JSONL input")
    infer.add_argument("--output-dir")
    infer.add_argument("--text")
    infer.add_argument("--reference")
    infer.add_argument("--reference-text")
    infer.add_argument("--output")
    infer.add_argument("--id")
    infer.add_argument("--language", default="tr")
    infer.add_argument("--emotion", default="neutral")
    infer.add_argument("--intensity", type=float, default=0.0)
    infer.add_argument("--delivery", default="neutral")
    infer.add_argument("--valence", type=float)
    infer.add_argument("--arousal", type=float)
    infer.add_argument(
        "--event",
        action="append",
        type=_json_value,
        help='repeatable JSON, e.g. \'{"type":"sob","after_word":2}\'',
    )
    infer.add_argument("--target-duration", type=float)
    infer.add_argument("--seed", type=int)
    infer.add_argument("--long-form", action="store_true")
    infer.set_defaults(handler=_infer)

    evaluate = subparsers.add_parser("evaluate", help="offline audio quality report")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--listening-manifest")
    evaluate.add_argument("--no-asr", action="store_true")
    evaluate.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--dtype", default="bfloat16")
    evaluate.add_argument("--batch-size", type=int, default=4)
    evaluate.add_argument("--language")
    evaluate.set_defaults(handler=_evaluate)

    audit = subparsers.add_parser("audit-data", help="filter a speech JSONL manifest")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--accepted", required=True)
    audit.add_argument("--rejected", required=True)
    audit.add_argument("--text-column", default="text")
    audit.add_argument("--minimum-seconds", type=float, default=0.4)
    audit.add_argument("--maximum-seconds", type=float, default=30.0)
    audit.add_argument("--maximum-clipping-ratio", type=float, default=0.001)
    audit.add_argument("--maximum-silence-ratio", type=float, default=0.75)
    audit.add_argument("--require-license", action="store_true")
    audit.set_defaults(handler=_audit_data)

    distill = subparsers.add_parser(
        "distill", help="build SFT data from verified winners"
    )
    distill.add_argument("--reports", nargs="+", required=True)
    distill.add_argument("--output", required=True)
    distill.add_argument("--minimum-score", type=float, default=0.75)
    distill.add_argument("--maximum-cer", type=float, default=0.10)
    distill.set_defaults(handler=_distill)

    expressive = subparsers.add_parser(
        "encode-expressive", help="encode controlled SFT JSONL and retain metadata"
    )
    expressive.add_argument("--manifest", required=True)
    expressive.add_argument("--output", required=True)
    expressive_model = expressive.add_mutually_exclusive_group(required=True)
    expressive_model.add_argument("--tokenizer")
    expressive_model.add_argument("--checkpoint")
    expressive.add_argument("--codec", default="kyutai/mimi")
    expressive.add_argument("--device", default="cuda:0")
    expressive.add_argument("--dtype", default="bfloat16")
    expressive.add_argument("--batch-size", type=int, default=16)
    expressive.add_argument("--tag-neutral", action="store_true")
    expressive.set_defaults(handler=_encode_expressive)

    codec = subparsers.add_parser(
        "codec-check", help="measure Mimi reconstruction before expressive training"
    )
    codec.add_argument("--audio", nargs="+", required=True)
    codec.add_argument("--output", required=True)
    codec.add_argument("--codec", default="kyutai/mimi")
    codec.add_argument("--device", default="cuda:0")
    codec.add_argument("--dtype", default="bfloat16")
    codec.add_argument("--num-codebooks", type=int, default=32)
    codec.add_argument("--batch-size", type=int, default=8)
    codec.set_defaults(handler=_codec_check)

    preprocess = subparsers.add_parser(
        "preprocess", help="encode raw speech into model-ready Parquet"
    )
    preprocess.add_argument(
        "--config",
        default=os.fspath(
            Path(__file__).resolve().parent / "configs" / "train" / "preprocess.yaml"
        ),
    )
    preprocess.set_defaults(handler=_preprocess)

    inspect = subparsers.add_parser(
        "inspect-tokenizer", help="print token IDs to copy into train configs"
    )
    inspect.add_argument("--model", required=True)
    inspect.add_argument("--revision")
    inspect.add_argument("--trust-remote-code", action="store_true")
    inspect.set_defaults(handler=_inspect_tokenizer)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (FileNotFoundError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
