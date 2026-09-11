"""Machine Failure Early Warning - Streamlit prototype.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import load_config, path, use_project_root  # noqa: E402
from src.features import RAW_NUMERIC  # noqa: E402
from src.ingest import ingest, read_source  # noqa: E402
from src.modeling import FAMILY_LABELS  # noqa: E402
from src.scoring import likely_mode, load_champion, prepare, registry_table, score, shift_outcome  # noqa: E402
from src.simulate import make_glitch_shift  # noqa: E402
from src.validate import validate_file  # noqa: E402

use_project_root()
cfg = load_config()
st.set_page_config(page_title="Machine Failure Early Warning", page_icon="🛠️", layout="wide")


@st.cache_resource(show_spinner="Loading the champion model from the MLflow Model Registry...")
def get_model():
    return load_champion("champion")


try:
    model, info = get_model()
except Exception as e:
    st.error(f"Could not load the champion model from the MLflow registry: {e}. "
             "Run 3_run_experiments.command first.")
    st.stop()

threshold = info["threshold"]
tags = info["tags"]

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Champion model")
    st.markdown(f"**{info['name']}** v{info['version']}  \n"
                f"{FAMILY_LABELS.get(tags.get('model_family'), tags.get('model_family'))}, "
                f"{tags.get('feature_set')} features, imbalance = {tags.get('imbalance')}")
    c1, c2 = st.columns(2)
    c1.metric("Alert threshold", f"{threshold:.2f}")
    c2.metric("Hold-out PR-AUC", tags.get("holdout_pr_auc", "-")[:5])
    c1.metric("Hold-out recall", tags.get("holdout_recall", "-")[:4])
    c2.metric("Hold-out precision", tags.get("holdout_precision", "-")[:4])
    st.caption(f"Trained on data {tags.get('data_version', '?')}. An alert is raised when the predicted "
               f"failure probability is at least {threshold:.2f}. The threshold was chosen to minimize expected "
               f"cost (missed failure ${cfg['costs']['missed_failure']:,}, planned repair "
               f"${cfg['costs']['caught_failure']:,}, false alarm ${cfg['costs']['false_alarm']:,}) while keeping "
               f"precision at least {cfg['targets']['min_precision']:.0%}.")
    if st.button("Reload model from registry"):
        st.cache_resource.clear()
        st.rerun()

st.title("🛠️ Machine Failure Early Warning")
st.caption("AI4I 2020 milling data · batch ingestion → Great Expectations validation → champion model from the MLflow registry")

tab_shift, tab_one, tab_reg = st.tabs(["Shift risk board", "Check one machine", "Model registry"])

# ----------------------------------------------------------------------------- tab 1: shift risk board
with tab_shift:
    raw_dir = path(cfg["data"]["raw_dir"])
    total_shifts = -(-len(read_source()) // cfg["data"]["shift_size"])
    landed = sorted(p.name for p in raw_dir.glob("shift_*.csv"))
    regular = [f for f in landed if f[6:9].isdigit() and int(f[6:9]) <= total_shifts]

    a, b, _ = st.columns([1, 1, 2])
    if a.button("⬇️ Ingest next shift", disabled=len(regular) >= total_shifts,
                help="Lands the next 500 runs from the replayed sensor feed into the raw zone"):
        new = ingest(len(regular) + 1)
        if new:
            st.session_state["pick"] = new[-1]["file"]
        st.rerun()
    if b.button("⚠️ Simulate sensor glitch",
                help="Writes a corrupted shift file to show validation catching bad data"):
        st.session_state["pick"] = make_glitch_shift()
        st.rerun()

    landed = sorted((p.name for p in raw_dir.glob("shift_*.csv")), reverse=True)
    default = st.session_state.get("pick", landed[0] if landed else None)
    choice = st.selectbox("Shift to score", landed, index=landed.index(default) if default in landed else 0,
                          help=f"Shifts 1-{cfg['data']['training_shifts']} were used for training history; "
                               "later shifts are new data the model has never seen.")
    if choice:
        n = int(choice[6:9]) if choice[6:9].isdigit() else 0
        if 0 < n <= cfg["data"]["training_shifts"]:
            st.info(f"{choice} is part of the training history. Shifts after "
                    f"{cfg['data']['training_shifts']} are unseen data.")
        report = validate_file(raw_dir / choice, cfg)
        gxr = report.get("great_expectations", {})

        st.subheader("1 · Validation")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Status", report["status"].replace("_", " "))
        v2.metric("Rows passed", report["rows_valid"])
        v3.metric("Rows quarantined", report["rows_quarantined"])
        v4.metric("GX expectations failed", f"{gxr.get('failed', '-')} / {gxr.get('evaluated', '-')}")
        with st.expander("Validation details"):
            rules = {k: v for k, v in report.get("row_rules", {}).items() if v}
            st.write("Row rules broken:", rules or "none")
            st.dataframe(pd.DataFrame(report.get("batch_checks", [])), hide_index=True, use_container_width=True)
            failed = [r for r in gxr.get("results", []) if not r["success"]]
            if failed:
                st.write("Failed Great Expectations checks:")
                st.dataframe(pd.DataFrame(failed), hide_index=True, use_container_width=True)
            q = path(cfg["data"]["quarantine_dir"]) / f"{Path(choice).stem}.csv"
            if report["rows_quarantined"] and q.exists() and q.stat().st_size:
                st.write("Quarantined rows (held for review, not scored):")
                st.dataframe(pd.read_csv(q).head(50), hide_index=True, use_container_width=True)

        st.subheader("2 · Failure risk")
        good = pd.read_parquet(path(cfg["data"]["validated_dir"]) / f"{Path(choice).stem}.parquet")
        if good.empty:
            st.warning("No rows passed validation, so nothing was scored.")
        else:
            scored = score(model, prepare(good), threshold)
            scored["likely_driver"] = scored.apply(likely_mode, axis=1)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Runs scored", len(scored))
            k2.metric("Alerts raised", int(scored["alert"].sum()))
            out = shift_outcome(scored)
            if out:
                k3.metric("Real failures caught", f"{out['caught']} of {out['failures']}")
                k4.metric("False alarms", out["false_alarms"])
            show = scored.head(15)[["product_id", "type", "failure_probability", "alert", "likely_driver",
                                    *RAW_NUMERIC] + (["machine_failure"] if "machine_failure" in scored else [])]
            st.dataframe(
                show, hide_index=True, use_container_width=True,
                column_config={
                    "failure_probability": st.column_config.ProgressColumn("failure probability", min_value=0.0,
                                                                           max_value=1.0, format="%.2f"),
                    "alert": st.column_config.CheckboxColumn("alert"),
                    "machine_failure": st.column_config.NumberColumn("actual failure (label)"),
                })
            st.caption("Top 15 runs in the shift ranked by predicted failure probability. "
                       "'actual failure' is the dataset label, shown only to check the model during the demo.")

# ----------------------------------------------------------------------------- tab 2: one machine
with tab_one:
    st.write("Enter current sensor readings for one machine run.")
    c1, c2, c3 = st.columns(3)
    ptype = c1.selectbox("Product quality type", ["L", "M", "H"])
    air = c1.slider("Air temperature [K]", 295.0, 305.0, 300.0, 0.1)
    proc = c2.slider("Process temperature [K]", 305.0, 314.0, 310.0, 0.1)
    rpm = c2.slider("Rotational speed [rpm]", 1150, 2900, 1500, 10)
    torque = c3.slider("Torque [Nm]", 3.0, 77.0, 40.0, 0.5)
    wear = c3.slider("Tool wear [min]", 0, 260, 100, 1)
    one = prepare(pd.DataFrame([{"type": ptype, "air_temp_k": air, "process_temp_k": proc,
                                 "rot_speed_rpm": rpm, "torque_nm": torque, "tool_wear_min": wear}]))
    res = score(model, one, threshold).iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Failure probability", f"{res['failure_probability']:.1%}")
    m2.metric("Temp. difference", f"{res['temp_diff_k']:.1f} K")
    m3.metric("Power", f"{res['power_w']:,.0f} W")
    m4.metric("Strain (wear x torque)", f"{res['strain_min_nm']:,.0f}")
    if res["alert"]:
        st.error(f"ALERT: schedule an inspection. Likely driver: {likely_mode(res)}")
    else:
        st.success(f"No alert (below the {threshold:.2f} threshold).")
    st.caption("Try: rotational speed under 1,380 rpm with a small temperature gap (heat dissipation), "
               "or high torque with tool wear over 200 minutes (overstrain).")

# ----------------------------------------------------------------------------- tab 3: registry
with tab_reg:
    st.write(f"Every version of **{info['name']}** in the MLflow Model Registry. "
             "The app always serves the version with the **champion** alias.")
    try:
        st.dataframe(registry_table(), hide_index=True, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not read the registry: {e}")
    lb = path("reports/leaderboard.csv")
    if lb.exists():
        st.write("Experiment leaderboard (top 10 by cross-validated PR-AUC):")
        st.dataframe(pd.read_csv(lb).head(10), hide_index=True, use_container_width=True)
    sel = path("reports/model_selection.json")
    if sel.exists():
        with st.expander("Selection summary (reports/model_selection.json)"):
            st.json(json.loads(sel.read_text()))
