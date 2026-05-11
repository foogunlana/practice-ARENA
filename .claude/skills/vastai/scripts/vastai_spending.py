#!/usr/bin/env python3
"""Show Vast.ai spending for the current month and running instances."""
import subprocess, json, sys
from datetime import datetime, timezone

def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"Error: {r.stderr.strip()}", file=sys.stderr)
        return None
    return r.stdout.strip()

def get_credit():
    out = run("vastai show user --raw")
    if not out: return None
    return json.loads(out).get("credit", 0)

def get_monthly_spend():
    now = datetime.now(timezone.utc)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    out = run(f"vastai show invoices --start_date '{start.strftime('%Y-%m-%d')}' --only_charges --raw")
    if not out: return 0.0
    try:
        invoices = json.loads(out)
        return sum(abs(i.get("amount", i.get("amount_cents", 0) / 100)) for i in invoices)
    except (json.JSONDecodeError, TypeError):
        return 0.0

def get_running_instances():
    out = run("vastai show instances --raw 2>/dev/null")
    if not out: return []
    try:
        return json.loads(out)
    except (json.JSONDecodeError, TypeError):
        return []

def main():
    credit = get_credit()
    month_spend = get_monthly_spend()
    instances = get_running_instances()
    budget = 10.0

    now = datetime.now(timezone.utc)
    print(f"=== Vast.ai Spending Report ({now.strftime('%Y-%m-%d %H:%M UTC')}) ===")
    print(f"Credit balance:  ${credit:.2f}" if credit is not None else "Credit balance:  (unavailable)")
    print(f"Month spend:     ${month_spend:.2f} / ${budget:.2f} ({month_spend/budget*100:.0f}%)")
    remaining = budget - month_spend
    print(f"Budget left:     ${remaining:.2f}")
    if remaining < 2:
        print("⚠️  WARNING: Less than $2 budget remaining this month!")

    if instances:
        running = [i for i in instances if i.get("actual_status") == "running"]
        stopped = [i for i in instances if i.get("actual_status") != "running"]
        if running:
            print(f"\nRunning instances ({len(running)}):")
            for i in running:
                hrs = (i.get("client_run_time", 0) or 0) / 3600
                print(f"  ID {i['id']} | {i.get('gpu_name','?')} | ${i.get('dph_total',0):.3f}/hr | {hrs:.1f}h runtime")
        if stopped:
            print(f"\nStopped instances ({len(stopped)}):")
            for i in stopped:
                print(f"  ID {i['id']} | {i.get('gpu_name','?')} | storage: ${i.get('storage_total_cost',0):.4f}/hr")
    else:
        print("\nNo active instances.")

if __name__ == "__main__":
    main()
