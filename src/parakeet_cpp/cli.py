"""Installed native Parakeet command line interface."""

import argparse
import sys

from . import Parakeet


def _parser():
    parser = argparse.ArgumentParser(
        prog="parakeet-cli", description="Transcribe WAV, MP3, FLAC, and OGG files."
    )
    parser.add_argument("files", nargs="*", help="input audio file")
    parser.add_argument("-f", "--file", dest="flag_files", action="append", default=[], help="input audio file")
    parser.add_argument("-t", "--threads", type=int, help="number of compute threads")
    parser.add_argument("-m", "--model", help="model path")
    parser.add_argument("-ng", "--no-gpu", action="store_true", help="disable GPU")
    parser.add_argument("-dev", "--device", type=int, default=0, help="GPU device")
    parser.add_argument("-ps", "--print-segments", action="store_true", help="print segments and tokens")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    files = args.files + args.flag_files
    if not files:
        _parser().error("no input files specified")
    try:
        parakeet = Parakeet(
            args.model,
            use_gpu=not args.no_gpu,
            gpu_device=args.device,
            n_threads=args.threads,
        )
        for path in files:
            result = parakeet.transcribe(path, print_segments=args.print_segments)
            print(result["text"])
            if args.print_segments:
                for segment in result["segments"]:
                    print('Segment {index}: [{t0} -> {t1}] "{text}"'.format(**segment), file=sys.stderr)
                    for token in segment["tokens"]:
                        print("  {index}: {text}".format(**token), file=sys.stderr)
    except (OSError, RuntimeError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
