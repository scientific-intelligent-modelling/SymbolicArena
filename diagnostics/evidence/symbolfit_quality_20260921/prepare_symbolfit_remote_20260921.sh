#!/bin/bash
set -eu
mkdir -p /tmp/symbolfit_debug_20260921/before /tmp/symbolfit_debug_20260921/after
tar -xzf /tmp/symbolfit_debug_pkg_20260921.tar.gz -C /tmp/symbolfit_debug_20260921/before
cp -a /tmp/symbolfit_debug_20260921/before/. /tmp/symbolfit_debug_20260921/after/
cp /tmp/symbolfit_quality_probe_20260921.py /tmp/symbolfit_debug_20260921/before/diagnostics/symbolfit_quality_probe.py
cp /tmp/symbolfit_quality_probe_20260921.py /tmp/symbolfit_debug_20260921/after/diagnostics/symbolfit_quality_probe.py
cp /tmp/symbolfit_fixed_wrapper_20260921.py /tmp/symbolfit_debug_20260921/after/scientific_intelligent_modelling/algorithms/symbolfit_wrapper/wrapper.py
