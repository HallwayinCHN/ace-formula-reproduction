"""Run the unchanged frozen evaluator through the separately sealed 503 recovery adapter."""

import argparse

from repro import holdout
from repro.recovery import OverloadAwareReplay, ProbeFinished, ServiceUnavailable
from repro.transport import ExperimentStopped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=holdout.VARIANTS, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    holdout.BalanceAwareReplay = lambda *values, **options: OverloadAwareReplay(
        *values, probe=args.probe, **options)
    holdout.run_job(args.variant, args.seed)


if __name__ == "__main__":
    try:
        main()
    except ProbeFinished as exc:
        print(f"PROBE_COMPLETED: {exc}")
        raise SystemExit(3)
    except ServiceUnavailable as exc:
        print(f"OVERLOAD_COOLDOWN: {exc}")
        raise SystemExit(75)
    except ExperimentStopped as exc:
        print(f"STOPPED: {exc}")
        raise SystemExit(2)
