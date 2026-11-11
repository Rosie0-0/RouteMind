"""
RouteMind - Step 8: the agent loop.

Per order:
  1. RF risk score (predict.py)                      -> prior
  2. Send-to-human check (BEFORE any reasoning)
  3. Bayesian update with SIMULATED truck evidence   -> posterior
  4. Fuzzy safety rules (posterior + SIMULATED stock) -> urgency -> action
  5. Reasoning card, human approval gate, feedback log
  6. SIMULATED tool call with retry -> backup -> escalate, and a loop limit
  7. Every decision logged as one JSON line; outcome appended to the
     incident log (stand-in for long-term memory)

All numbers come from the `agent` section of config.yaml.

  python agent_demo.py --scripted        run all scenarios with scripted answers
  python agent_demo.py --only A          run one scenario and type the answers
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime

from features import load_config
from predict import RiskPredictor

OUT = []          # lines of the run log


def say(text=""):
    print(text)
    OUT.append(text)


def append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------- Bayesian update -------------------------------------------------
def bayes_update(prior, idle, cfg):
    """posterior = P(E|late)*p / (P(E|late)*p + P(E|on time)*(1-p))"""
    b = cfg["bayes"]
    lo, hi = b["prior_clamp"]
    p = min(max(prior, lo), hi)
    if idle is None:                       # no evidence -> belief unchanged
        return p, round(p, b["posterior_decimals"])
    lk = b["idle_true"] if idle else b["idle_false"]
    l1, l0 = lk["p_given_late"], lk["p_given_on_time"]
    post = l1 * p / (l1 * p + l0 * (1 - p))
    return p, round(post, b["posterior_decimals"])


# ---------- Fuzzy logic -----------------------------------------------------
def membership(x, shape, pts):
    """How strongly x belongs to a fuzzy set, from 0 (not at all) to 1 (fully)."""
    a, b, c, d = pts if shape == "trap" else (pts[0], pts[1], pts[1], pts[2])
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def fuzzy_decide(posterior, stock, f):
    """f = the `agent.fuzzy` section of config.yaml."""
    risk_m = {k: membership(posterior, v["shape"], v["points"]) for k, v in f["risk"].items()}
    stock_m = {k: membership(stock, v["shape"], v["points"]) for k, v in f["stock_days"].items()}
    fired = {}
    for name, (r, s, action) in f["rules"].items():
        strength = min(risk_m[r], stock_m[s])          # AND = minimum
        if strength > 0:
            fired[name] = {"if": f"risk {r} AND stock {s}", "then": action,
                           "strength": round(strength, 2)}
    total = sum(min(risk_m[r], stock_m[s]) for r, s, _ in f["rules"].values())
    if total == 0:
        return risk_m, stock_m, fired, None, None
    urgency = sum(min(risk_m[r], stock_m[s]) * f["outputs"][a]
                  for r, s, a in f["rules"].values()) / total
    urgency = round(urgency + 1e-9, f["urgency_decimals"])   # round BEFORE banding
    action = next(a for limit, a in f["bands"] if urgency < limit)
    return risk_m, stock_m, fired, urgency, action


# ---------- Memory stand-in -------------------------------------------------
def similar_cases(order, cfg):
    path = cfg["logs"]["incidents"]
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    same = [r for r in rows if r.get("shipping_mode") == order.get("Shipping Mode")]
    return same[-cfg["logs"]["similar_cases_shown"]:]


# ---------- Reasoning card --------------------------------------------------
def reasoning_card(state, cfg, request):
    say("  +---------------- REASONING CARD ----------------")
    say(f"  | Order: {state['order_id']} | Shipping Mode: {state['shipping_mode']}")
    say(f"  | RF risk score (prior): {state['prior_text']}")
    if state.get("human_reasons"):
        for r in state["human_reasons"]:
            say(f"  | STOPPED: {r}")
    if state.get("posterior") is not None:
        say(f"  | Evidence (SIMULATED): truck idle > 60 min = {state['idle']}")
        say(f"  | Posterior after Bayes: {state['posterior']:.2f}")
    if state.get("risk_m"):
        say("  | Risk membership:  " + ", ".join(f"{k} {v:.2f}" for k, v in state["risk_m"].items()))
        say(f"  | Stock (SIMULATED) = {state['stock']} days: "
            + ", ".join(f"{k} {v:.2f}" for k, v in state["stock_m"].items()))
        for name, r in state["fired"].items():
            say(f"  | {name}: IF {r['if']} THEN {r['then']}  (strength {r['strength']:.2f})")
        say(f"  | Urgency: {state['urgency']}  ->  proposed action: {state['action']}")
    say("  | Top features (global importances, same for every order): "
        + ", ".join(f"{n.replace('_', ' ')} {s:.0%}" for n, s in state["top_features"]))
    cases = state.get("similar", [])
    if cases:
        say("  | Similar past cases (incident log, same shipping mode):")
        for c in cases:
            say(f"  |   {c['time']} {c['order_id']}: {c['action']} -> {c['outcome']}")
    else:
        say("  | Similar past cases: none recorded yet")
    say(f"  | {request}")
    say("  +------------------------------------------------")


# ---------- Human input -----------------------------------------------------
def ask(prompt, scripted_value, scripted):
    if scripted:
        say(f"  {prompt} [scripted answer: {scripted_value}]")
        return str(scripted_value)
    answer = input(f"  {prompt} ").strip()
    OUT.append(f"  {prompt} [typed: {answer}]")
    return answer


# ---------- SIMULATED tool call --------------------------------------------
def call_reroute(tool, cfg):
    """Returns (success, trace). Primary API -> check booking -> retry with
    backoff -> backup API -> give up (caller escalates)."""
    t = cfg["tools"]
    responses = list(tool.get("primary", ["ok"]))
    trace = []
    for attempt in range(t["max_retries"] + 1):
        if attempt > 0:
            wait = t["backoff_seconds"][attempt - 1]
            trace.append(f"retry {attempt}/{t['max_retries']} after {wait}s backoff (SIMULATED wait)")
        result = responses[attempt] if attempt < len(responses) else "timeout"
        trace.append(f"primary routing API -> {result} (SIMULATED)")
        if result == "ok":
            return True, trace
        # A timeout does not prove the reroute failed: check before retrying,
        # so the same shipment is never rerouted twice.
        status = tool.get("booking_status", "not_booked")
        trace.append(f"check booking status -> {status} (SIMULATED)")
        if status == "booked":
            return True, trace
    backup = tool.get("backup", "ok")
    trace.append(f"backup routing API -> {backup} (SIMULATED)")
    return backup == "ok", trace


# ---------- One scenario ----------------------------------------------------
def run_scenario(sc, orders, predictor, cfg, scripted, decision_log):
    a = cfg["agent"]
    order = orders[sc["order"]]
    answers = sc.get("answers", {})
    say("=" * 66)
    say(f"SCENARIO {sc['id']}: {sc['title']}")
    say("=" * 66)

    pred = predictor.score(order)
    prior = sc.get("rf_prior_override", pred["risk"])
    state = {"scenario": sc["id"], "order_id": pred["order_id"],
             "shipping_mode": order.get("Shipping Mode"), "rf_risk": pred["risk"],
             "prior": prior, "idle": sc.get("idle"), "stock": sc.get("stock_days"),
             "top_features": pred["top_features"], "similar": similar_cases(order, a),
             "posterior": None, "final_action": None, "human_response": None,
             "tool_trace": [], "reroutes": 0}
    state["prior_text"] = (f"{prior:.2f} (worked-example prior from Task 1 Section 1.3; "
                           f"model gave {pred['risk']:.3f})"
                           if "rf_prior_override" in sc else f"{pred['risk']:.3f}")

    def finish(action, outcome):
        state["final_action"], state["outcome"] = action, outcome
        say(f"  RESULT: {action} - {outcome}")
        record = {k: v for k, v in state.items()
                  if k not in ("top_features", "similar", "prior_text")}
        record["time"] = datetime.now().isoformat(timespec="seconds")
        append_jsonl(decision_log, record)
        append_jsonl(a["logs"]["incidents"], {
            "time": record["time"], "order_id": state["order_id"],
            "shipping_mode": state["shipping_mode"], "posterior": state["posterior"],
            "action": action, "outcome": outcome})
        return state

    # Step 2: send-to-human check comes BEFORE Bayes and fuzzy
    if pred["route_to_human"]:
        state["human_reasons"] = pred["human_reasons"]
        reasoning_card(state, a, "No automatic action. A human must review this order.")
        return finish("SendToHuman", "stopped before reasoning: " + "; ".join(pred["human_reasons"]))

    followups = list(sc.get("followup_idle", []))
    while True:
        # Step 3: Bayes
        _, state["posterior"] = bayes_update(prior, state["idle"], a)
        say(f"  RF prior {prior:.2f} | evidence idle={state['idle']} (SIMULATED) "
            f"| posterior {state['posterior']:.2f}")

        # Step 4: fuzzy (needs both inputs)
        if state["stock"] is None or state["posterior"] is None:
            state["human_reasons"] = ["stock or risk input missing: fuzzy rules skipped"]
            reasoning_card(state, a, "No automatic action. A human must review this order.")
            return finish("SendToHuman", "missing input for the fuzzy rules")
        (state["risk_m"], state["stock_m"], state["fired"],
         state["urgency"], state["action"]) = fuzzy_decide(state["posterior"], state["stock"], a["fuzzy"])
        if state["action"] is None:
            state["human_reasons"] = ["no fuzzy rule fired"]
            reasoning_card(state, a, "No automatic action. A human must review this order.")
            return finish("SendToHuman", "no fuzzy rule fired")
        say("  Rules fired: " + ", ".join(f"{k} {v['strength']:.2f}" for k, v in state["fired"].items())
            + f" | urgency {state['urgency']} | action {state['action']}")
        action = state["action"]

        if action in ("Continue", "Monitor"):
            return finish(action, "no intervention; order stays under observation")
        if action == "Escalate":
            reasoning_card(state, a, "Escalated to the human supervisor (no automatic action).")
            return finish("Escalate", "handed to human supervisor")

        # Reroute: loop limit first, then approval gate
        limit = a["tools"]["max_reroutes_per_order"]
        if state["reroutes"] >= limit:
            say(f"  LOOP LIMIT: {state['reroutes']} reroutes already made for this order.")
            reasoning_card(state, a, f"Loop limit ({limit}) reached. Escalated to the human supervisor.")
            return finish("Escalate", f"loop limit reached after {state['reroutes']} reroutes")

        reasoning_card(state, a, "APPROVAL NEEDED: reroute this shipment? (y/n)")
        reply = ask("Approve reroute? (y/n)", answers.get("approve", "y"), scripted).lower()
        if reply != "y":
            reasons = a["reject_reasons"]
            for code, text in reasons.items():
                say(f"    {code}. {text}")
            code = ask("Reason for rejecting (1-6)?", answers.get("reason", "6"), scripted)
            text = reasons.get(code, reasons["6"])
            if code == "6" or code not in reasons:
                text += ": " + ask("Please describe:", answers.get("reason_text", "not given"), scripted)
            state["human_response"] = {"approved": False, "reason_code": code, "reason": text}
            append_jsonl(a["logs"]["feedback"], {
                "time": datetime.now().isoformat(timespec="seconds"),
                "order_id": state["order_id"], "proposed_action": "Reroute",
                "posterior": state["posterior"], "urgency": state["urgency"],
                "reason_code": code, "reason": text})
            say(f"  Feedback saved to {a['logs']['feedback']}")
            return finish("Monitor", f"reroute rejected by human ({text}); kept under observation")
        state["human_response"] = {"approved": True}

        # Step 7: SIMULATED tool call
        ok, trace = call_reroute(sc.get("tool", {}), a)
        state["tool_trace"] += trace
        for line in trace:
            say(f"    tool: {line}")
        if not ok:
            reasoning_card(state, a, "Routing tools unavailable. Escalated to the human supervisor.")
            return finish("Escalate", "primary and backup routing API failed")
        state["reroutes"] += 1
        if not followups:
            return finish("Reroute", f"shipment rerouted (reroute {state['reroutes']})")
        state["idle"] = followups.pop(0)
        say(f"  Reroute {state['reroutes']} done. New observation (SIMULATED): idle={state['idle']}")


def check(sc, state):
    exp = sc.get("expect")
    if not exp:
        return None
    problems = []
    got_action = state["final_action"] if exp["action"] == "SendToHuman" else state.get("action")
    if got_action != exp["action"]:
        problems.append(f"action {got_action} != {exp['action']}")
    if "posterior" in exp and state["posterior"] != exp["posterior"]:
        problems.append(f"posterior {state['posterior']} != {exp['posterior']}")
    if "urgency" in exp and state.get("urgency") != exp["urgency"]:
        problems.append(f"urgency {state.get('urgency')} != {exp['urgency']}")
    if "rules" in exp:
        got = {k: v["strength"] for k, v in state.get("fired", {}).items()}
        if got != {k: round(v, 2) for k, v in exp["rules"].items()}:
            problems.append(f"rules {got} != {exp['rules']}")
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scripted", action="store_true",
                        help="use the scripted answers in agent_scenarios.json")
    parser.add_argument("--only", help="run one scenario id, e.g. A")
    args = parser.parse_args()

    cfg = load_config()
    a = cfg["agent"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = cfg["paths"]["log_dir"]
    decision_log = os.path.join(log_dir, f"decisions_{stamp}.jsonl")

    predictor = RiskPredictor(cfg)
    with open(cfg["prediction"]["sample_orders"], encoding="utf-8") as f:
        orders = {o["order_id"]: o for o in json.load(f)}
    with open(a["scenarios"], encoding="utf-8") as f:
        scenarios = [s for s in json.load(f) if not args.only or s["id"] == args.only]

    say("ROUTEMIND AGENT DEMO  (evidence, stock and tool calls are SIMULATED)")
    say(f"Mode: {'scripted' if args.scripted else 'interactive'} | threshold "
        f"{predictor.threshold} | band +/-{predictor.band}")

    results = []
    for sc in scenarios:
        state = run_scenario(sc, orders, predictor, cfg, args.scripted, decision_log)
        problems = check(sc, state)
        if problems is not None:
            verdict = "PASS" if not problems else "FAIL: " + "; ".join(problems)
            say(f"  TEST {sc['id']}: {verdict}")
            results.append((sc["id"], not problems))
        say()

    if results:
        passed = sum(ok for _, ok in results)
        say(f"TEST SUMMARY: {passed}/{len(results)} passed  "
            + " ".join(f"{i}:{'PASS' if ok else 'FAIL'}" for i, ok in results))
    run_log = os.path.join(log_dir, f"agent_{stamp}.log")
    with open(run_log, "w", encoding="utf-8") as f:
        f.write("\n".join(OUT))
    print(f"Run log: {run_log}\nDecision log: {decision_log}")
    sys.exit(0 if all(ok for _, ok in results) else 1)


if __name__ == "__main__":
    main()
