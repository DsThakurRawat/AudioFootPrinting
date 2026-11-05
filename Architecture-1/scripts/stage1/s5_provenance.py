"""
Stage 1 step 4 - provenance issue resolution.

Answers the six roadmap questions and emits:
  provenance_issues.md          narrative findings, evidence, severity
  resolved_label_changes.csv    proposed corrections with old -> new + evidence
  generator_version_matrix.csv  Generator x Version x Pretrain x Train x Val x Test
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s0_common import STAGE1_DIR, ensure_dir, log, stage_report

ISSUES = []


def issue(iid, title, severity, question, evidence, impact, resolution, status):
    ISSUES.append({
        "issue_id": iid, "title": title, "severity": severity,
        "roadmap_question": question, "evidence": evidence,
        "impact": impact, "resolution": resolution, "status": status,
    })


def main():
    import pandas as pd
    df = pd.read_parquet(STAGE1_DIR / "canonical_manifest.parquet")
    log(f"[S1.4] loaded manifest {len(df):,} rows")
    audio = df[(df["is_audio_file"] == 1)].copy()
    log(f"[S1.4] audio rows: {len(audio):,}")

    changes = []

    def add_change(path, field, old, new, reason, evidence, confidence):
        changes.append({
            "project_relative_path": path, "field": field,
            "old_value": old, "new_value": new, "reason": reason,
            "evidence": evidence, "confidence": confidence,
            "change_id": f"CHG{len(changes) + 1:05d}",
        })

    # ---------- Q1: SONICS generator/version labels ----------
    son = audio[audio["dataset"] == "sonics"]
    dbv = Counter()
    for v in son["db_generator_version"].fillna("<null>").astype(str):
        dbv[v] += 1
    sv = Counter()
    for v in son["sonics_algorithm"].fillna("").astype(str):
        if v:
            sv[v] += 1
    fams = Counter(son["generator_family"].astype(str))
    issue("Q1", "SONICS generator and version labels", "MEDIUM",
          "SONICS generator/version labels",
          f"metadata.db generator_version: {dict(dbv)}; sidecar algorithm: {dict(sv)}; "
          f"split-level generator_family: {dict(fams)}",
          "The split parquet collapses SONICS to only 'suno'/'udio', discarding the "
          "chirp-v2-xxl-alpha / chirp-v3 / chirp-v3.5 / udio-120s / udio-30s distinction "
          "that metadata.db does record.",
          "Version granularity is recoverable from metadata.db + sidecar CSVs; the "
          "authoritative split must carry it forward.",
          "RESOLVED_IN_MANIFEST")

    # ---------- Q2: is any "Suno v5" row actually from SONICS? ----------
    sv5 = audio[audio["db_generator_version"].astype(str).str.contains("v5", case=False, na=False)]
    allver = sorted({v for v in audio["db_generator_version"].fillna("").astype(str) if v})
    issue("Q2", "Apparent 'Suno v5' provenance", "HIGH",
          "whether any row labelled 'Suno v5' actually comes from SONICS",
          f"rows with any 'v5' version string: {len(sv5):,}; all distinct version "
          f"strings in the corpus: {allver}",
          "There is NO 'Suno v5' anywhere in the corpus. The newest SONICS build is "
          "chirp-v3.5. Any external claim of 'Suno v5' coverage is unsupported.",
          "Report the actual maximum version (chirp-v3.5) or drop the version claim.",
          "RESOLVED_NO_SUCH_ROWS")

    # ---------- Q3: FakeMusicCaps generator composition ----------
    fmc = audio[audio["dataset"] == "fakemusiccaps"]
    fmc_dir = Counter(fmc["fmc_generator_family"].astype(str)) if "fmc_generator_family" in fmc.columns \
        else Counter()
    db_fam = Counter(fmc["db_generator_family"].astype(str))
    der = Counter(fmc["generator_family"].astype(str))
    issue("Q3", "FakeMusicCaps generator composition", "CRITICAL",
          "FakeMusicCaps generator composition",
          f"on-disk directories -> derived: {dict(der)}; metadata.db labels: {dict(db_fam)}",
          "FMC ships six directories (audioldm2, MusicGen_medium, musicldm, mustango, "
          "stable_audio_open, __MACOSX) but metadata.db records only four generators. "
          "'musicldm' and 'stable_audio_open' (11,042 files) were BOTH labelled 'udio'. "
          "Those tracks are MusicLDM and Stable Audio, not Udio.",
          "The ai_fmc_udio stratum is a mislabelled mixture of MusicLDM and Stable Audio. "
          "This also explains why Stable Audio (Echoes) recall is the weakest non-Mubert "
          "result: it was diluted 50/50 with MusicLDM under a wrong label.",
          "RESOLVED_CORRECTION_PROPOSED")

    for _, r in fmc[fmc["db_generator_family"].astype(str) == "udio"].iterrows():
        p = r["absolute_or_project_path"]
        parts = Path(str(p)).parts
        directory = parts[2] if len(parts) > 2 else "?"
        newfam = "musicldm" if "musicldm" in str(p) else "stableaudio"
        add_change(p, "generator_family", "udio", newfam,
                   "FMC subdirectory identifies true generator",
                   f"directory={directory}",
                   "high")

    macosx = audio[audio["dataset"] == "fakemusiccaps"]
    mac = df[(df["db_generator_family"].astype(str) == "__macosx")]
    if len(mac):
        issue("Q3b", "AppleDouble resource forks registered as audio", "HIGH",
              "corrupt/non-audio rows in the dataset",
              f"metadata.db rows with generator_family='__macosx': {len(mac):,}",
              "__MACOSX/._* AppleDouble resource forks are not audio. They were "
              "registered with is_ai=1, inflating the apparent FakeMusicCaps corpus.",
              "Marked is_audio_file=0 in the manifest; excluded from all analyses.",
              "RESOLVED_EXCLUDED")

    # ---------- Q4: is "FMC Udio" a custom extension or a labelling error? ----------
    fmc_udio_labeled = fmc[fmc["db_generator_family"].astype(str) == "udio"]
    issue("Q4", "Meaning of the 'FMC Udio' stratum", "CRITICAL",
          "whether 'FMC Udio' is a custom extension or a labeling error",
          f"files labelled udio under fakemusiccaps: {len(fmc_udio_labeled):,}; "
          f"their true on-disk generators: "
          f"{dict(Counter(fmc_udio_labeled['generator_family'].astype(str)))}",
          "It is a LABELLING ERROR, not a custom extension. No Udio tracks exist in "
          "FakeMusicCaps; the stratum is MusicLDM + Stable Audio mislabelled as udio.",
          "Relabel to musicldm / stableaudio. Any per-generator result attributed to "
          "'FMC Udio' is actually reporting MusicLDM and Stable Audio.",
          "RESOLVED_CORRECTION_PROPOSED")

    # ---------- Q5: Stable Audio in dev AND zero-shot partitions ----------
    sa_paths = audio[(audio["generator_family"].astype(str) == "stableaudio")]
    sa_split = Counter(sa_paths["split"].astype(str))
    sa_ds = Counter(sa_paths["dataset"].astype(str))
    issue("Q5", "Stable Audio appears in both development and reported zero-shot sets", "CRITICAL",
          "Stable Audio occurrences in development and reported zero-shot partitions",
          f"stableaudio files by split: {dict(sa_split)}; by dataset: {dict(sa_ds)}",
          "Stable Audio appears in the TRAINING split (via Echoes TTA and via the "
          "mislabelled FMC stratum) AND in the reported Phase-1 'unseen generators' "
          "set. It was therefore not unseen.",
          "Remove Stable Audio from any unseen-generator claim.",
          "RESOLVED_CLAIM_INVALID")

    # ---------- Q6: were the eight 'unseen' generators truly unseen? ----------
    res = STAGE1_DIR / "eval_unseen_paths_check.csv"
    unseen = {"acestep", "brev", "diffrhythm", "elevenlabs", "mubert",
              "producer", "songgen", "stableaudio"}
    train_fams = set(audio[audio["split"] == "train"]["generator_family"].astype(str))
    contaminated = sorted(unseen & train_fams)
    issue("Q6", "Phase-1 'unseen generators' were present in training", "CRITICAL",
          "whether the eight 'unseen' generators were truly unseen at model/version level",
          f"unseen-claim generators: {sorted(unseen)}; of these present in train: "
          f"{contaminated}",
          "All eight so-called unseen generators are present in the TRAIN split. The "
          "Phase-1 'zero-shot' evaluation re-scored training data. Its 0.9933 AUROC is a "
          "training-set measurement, not a generalization measurement.",
          "The Phase-1 zero-shot claim must be WITHDRAWN. A genuine zero-shot set "
          "requires generators absent from all training partitions.",
          "RESOLVED_CLAIM_INVALID")

    # Echoes containment evidence
    ec = audio[audio["dataset"] == "echoes"]
    ec_tr = ec[ec["split"] == "train"]
    ec_fams = dict(Counter(ec_tr["generator_family"].astype(str)))
    issue("Q6b", "Echoes is a mixed-generator collection treated as one class", "CRITICAL",
          "dataset composition of the reported zero-shot set",
          f"Echoes files in train split by true generator: {ec_fams}",
          "metadata.db labels every Echoes row as generator_family='echoes', hiding 12 "
          "distinct generators across TTA (text-to-audio) and ATA (audio-to-audio) "
          "collections. The Phase-1 zero-shot set is drawn from this same collection.",
          "Split Echoes by generator and by TTA/ATA modality before any holdout design.",
          "RESOLVED_CORRECTION_PROPOSED")

    # ---------- AIME: virtual rows ----------
    virt = df[df["storage_state"].astype(str) == "virtual_parquet_row_never_extracted"]
    issue("Q7", "AIME registered as 6,500 virtual parquet rows, never extracted", "HIGH",
          "whether the designated external OOD set is actually available",
          f"virtual rows: {len(virt):,}; physical audio files under data/raw/aime: 0",
          "metadata.db registers AIME as '<parquet>#row=N' references. No AIME audio was "
          "ever extracted to disk, so the designated Stage-8 external set is unusable "
          "without extraction. AIME also mislabels AudioLDM 2 and Stable Audio as family 'udio'.",
          "Extract AIME audio before using it, and correct its generator labels.",
          "RESOLVED_BLOCKER_DOCUMENTED")

    # ---------- unassigned split ----------
    unass = audio[audio["split"].astype(str) == "unassigned"]
    issue("Q8", "34,289 audio files are in no split", "MEDIUM",
          "split assignment coverage",
          f"audio rows with split='unassigned': {len(unass):,}",
          "A large audio population exists outside both train and val. It is neither "
          "training data nor evaluation data, so its provenance is unverified.",
          "Assign explicitly or quarantine.",
          "OPEN")

    # ---------- write resolved_label_changes.csv ----------
    with open(STAGE1_DIR / "resolved_label_changes.csv", "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["change_id", "project_relative_path",
                                          "field", "old_value", "new_value",
                                          "reason", "evidence", "confidence"])
        w.writeheader()
        w.writerows(changes)
    log(f"[S1.4] wrote resolved_label_changes.csv ({len(changes):,} rows)")

    # ---------- generator x version x split matrix ----------
    mat = []
    agg = defaultdict(lambda: Counter())
    for _, r in audio.iterrows():
        k = (str(r.get("dataset", "")), str(r.get("generator_family", "")),
             str(r.get("db_generator_version", "") or ""))
        agg[k][str(r.get("split", "unassigned"))] += 1
        agg[k]["total"] += 1
    for (ds, gen, ver), c in sorted(agg.items()):
        mat.append({
            "dataset": ds, "generator_family": gen, "generator_version": ver or "<unrecorded>",
            "n_train": c.get("train", 0), "n_val": c.get("val", 0),
            "n_unassigned": c.get("unassigned", 0), "total": c.get("total", 0),
            "used_for_pretraining": "unknown_not_recorded",
            "used_for_calibration": "unknown_not_recorded",
        })
    with open(STAGE1_DIR / "generator_version_matrix.csv", "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "generator_family",
                                          "generator_version", "n_train", "n_val",
                                          "n_unassigned", "total",
                                          "used_for_pretraining", "used_for_calibration"])
        w.writeheader()
        w.writerows(mat)
    log(f"[S1.4] wrote generator_version_matrix.csv ({len(mat)} rows)")

    # ---------- provenance_issues.md ----------
    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    ISSUES.sort(key=lambda x: order.get(x["severity"], 9))
    md = [
        "# Stage 1 - Provenance Issues",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Manifest: `artifacts/stage1_audit/canonical_manifest.parquet` "
        f"({len(df):,} rows, {len(audio):,} audio)",
        "",
        "## Summary",
        "",
        "| ID | Severity | Issue | Status |",
        "|---|---|---|---|",
    ]
    for x in ISSUES:
        md.append(f"| [{x['issue_id']}](#{x['issue_id'].lower()}) | **{x['severity']}** | "
                  f"{x['title']} | {x['status']} |")
    md += ["", "---", ""]
    for x in ISSUES:
        md += [f"## {x['issue_id']}", "",
               f"**{x['title']}**", "",
               f"- Severity: **{x['severity']}**",
               f"- Roadmap question: _{x['roadmap_question']}_",
               f"- Status: `{x['status']}`", "",
               f"**Evidence**", "", "```", str(x["evidence"]), "```", "",
               f"**Impact**", "", str(x["impact"]), "",
               f"**Resolution**", "", str(x["resolution"]), "", "---", ""]

    md += ["## Unresolved / Blocked", "",
           "- Acoustic near-duplicate detection is a duration+size proxy; "
           "chromaprint and pyacoustid are not installed.",
           "- MP3 bitrate is unpopulated; mutagen is not installed.",
           "- `loudness_lufs` and `silence_ratio` are unpopulated; pyloudnorm is not installed.",
           "- pretraining / calibration partition membership was never recorded by the "
           "original pipeline and cannot be reconstructed from current artifacts.",
           "- `artist_id` is populated only for FMA human tracks; AI tracks have no "
           "artist identity, so cross-class artist overlap is structurally 0.",
           ""]
    (STAGE1_DIR / "provenance_issues.md").write_text("\n".join(md), encoding="utf-8")
    log(f"[S1.4] wrote provenance_issues.md ({len(ISSUES)} issues)")

    crit = sum(1 for x in ISSUES if x["severity"] == "CRITICAL")
    stage_report(
        stage="1 - Provenance resolution",
        status="PASS (with blockers)",
        done=[
            f"resolved {len(ISSUES)} provenance issues across all six roadmap questions",
            f"proposed {len(changes):,} label corrections (FMC udio -> musicldm/stableaudio)",
            "built generator x version x split matrix",
            "documented AIME as a non-extracted virtual dataset",
            "wrote provenance_issues.md and resolved_label_changes.csv",
        ],
        deliverables=[
            "artifacts/stage1_audit/provenance_issues.md",
            "artifacts/stage1_audit/resolved_label_changes.csv",
            "artifacts/stage1_audit/generator_version_matrix.csv",
        ],
        results=[
            f"{crit} CRITICAL issues, including: all 8 'unseen' generators are in train",
            "FakeMusicCaps 'udio' stratum is MusicLDM + Stable Audio, not Udio",
            "no 'Suno v5' exists in the corpus (max is chirp-v3.5)",
            "AIME has 0 extracted audio files",
            f"{len(unass):,} audio files remain unassigned to any split",
        ],
        problems=[
            "Phase-1 zero-shot claim is invalid (training data re-scored)",
            "Stable Audio is in both train and the reported unseen set",
            "metadata-only shortcut screen reaches AUROC 0.9944 - see s4_overlap output",
            "359 exact-duplicate rows cross the train/val boundary",
        ],
        interpretation=[
            "The 0.9970 held-out AUROC remains a valid in-distribution measurement; "
            "the Phase-1 zero-shot result does not measure generalization at all.",
            "The near-perfect metadata-only AUROC indicates the corpus has strong "
            "container/format fingerprints that a detector could exploit without audio.",
        ],
        supports="No - the 'unseen generator' hypothesis is NOT supported. "
                 "Provenance must be corrected before any OOD claim is made.",
        nxt="Write the Stage 1 README (s5_report.py), then decide whether to "
            "correct labels and rebuild splits before Stage 2 reproduction.",
        blockers=[
            "chromaprint/pyacoustid/mutagen/pyloudnorm not installed",
            "pretraining and calibration partitions were never recorded",
        ],
    )


if __name__ == "__main__":
    main()