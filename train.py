#!/usr/bin/env python3
import argparse, src.training as T
ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
args = ap.parse_args()
T.fit(args.config)
