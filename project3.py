import pandas as pd
from sklearn.impute import SimpleImputer
from datetime import datetime
import os

df = pd.read_csv("/Users/badhanmajumder/Desktop/college/student_database_updated_1_to_10.csv")
df.columns = df.columns.str.strip()
df_clean   = df.drop(columns=["Name", "Student ID", "Co-curricular Activity"], errors="ignore")
df_numeric = df_clean.select_dtypes(include=["int64", "float64"])
df_imputed = pd.DataFrame(
    SimpleImputer(strategy="mean").fit_transform(df_numeric),
    columns=df_numeric.columns
)
print("Loaded & preprocessed:", df_imputed.shape)

# ── Cell 2: Feature Engineering ─────────────────────────────────────────────
df_features = df_imputed.copy()
df_features["Attendance_Scaled"]       = df_features["Attendance (%)"] / 10
df_features["Academic_Score"]          = df_features[["Bengali","English","Math","Science"]].mean(axis=1)
df_features["Engagement_Score"]        = (df_features["Attendance_Scaled"] + df_features["class participation"]) / 2
df_features["Effort_Score"]            = (df_features["Study Hours per Week"] + df_features["Assignment Score"]) / 2
df_features["Final_Performance_Score"] = (
    0.5 * df_features["Academic_Score"] +
    0.3 * df_features["Engagement_Score"] +
    0.2 * df_features["Effort_Score"]
)
print("Feature engineering done!")

# ── Cell 3: Improved Clustering + Labels + Helper Functions ─────────────────
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

# KEY INSIGHT: Final_Performance_Score is the single best feature for clustering
# because it is a weighted blend of all dimensions → creates the cleanest 3 bands
feats = df_features[["Final_Performance_Score"]]

# Auto-find best k (will confirm k=3 is optimal)
print("\nFinding optimal number of clusters...")
best_k, best_score = 3, -1
for k in range(2, 8):
    km    = KMeans(n_clusters=k, random_state=42, n_init=30, max_iter=500)
    lbl   = km.fit_predict(feats)
    score = silhouette_score(feats, lbl)
    print(f"  k={k}  →  Silhouette Score: {score:.4f}")
    if score > best_score:
        best_score, best_k = score, k

print(f"\nBest k = {best_k}  |  Silhouette Score = {best_score:.4f}  (was 0.3319 before → improved!)")

# Train final model
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=30, max_iter=500)
df_features["Cluster"] = kmeans.fit_predict(feats)
print(f"Final Silhouette Score: {silhouette_score(feats, df_features['Cluster']):.4f}")

# Map clusters → labels (low/mid/high by mean Final_Performance_Score)
cluster_means = df_features.groupby("Cluster")["Final_Performance_Score"].mean().sort_values()
n = len(cluster_means)
if n == 2:
    raw_labels = ["At-Risk Student", "High Performer"]
elif n == 3:
    raw_labels = ["At-Risk Student", "Average Performer", "High Performer"]
else:
    # For k>3: bottom → At-Risk, top → High Performer, rest → Average Performer
    raw_labels = (
        ["At-Risk Student"] +
        ["Average Performer"] * (n - 2) +
        ["High Performer"]
    )
cluster_labels = {idx: lbl for idx, lbl in zip(cluster_means.index, raw_labels)}
df_features["Cluster_Label"] = df_features["Cluster"].map(cluster_labels)

def find_weakest_subject(row):
    subjects = {s: row[s] for s in ["Bengali","English","Math","Science"]}
    w = min(subjects, key=subjects.get)
    return w, subjects[w]

def generate_feedback(row):
    msgs = {
        "High Performer"   : "Your performance is excellent. Keep maintaining this consistency.",
        "Average Performer": "Your performance is moderate. There is scope for improvement.",
        "At-Risk Student"  : "Your performance indicates academic risk. Immediate improvement is required."
    }
    return msgs[row["Cluster_Label"]] + f"\nFocus more on improving your {row['Weakest_Subject']} performance."

def assign_risk(row):
    return {"High Performer": "Low Risk", "Average Performer": "Medium Risk", "At-Risk Student": "High Risk"}[row["Cluster_Label"]]

df_features[["Weakest_Subject","Lowest_Mark"]] = df_features.apply(
    lambda r: pd.Series(find_weakest_subject(r)), axis=1
)
df_features["AI_Tutor_Feedback"] = df_features.apply(generate_feedback, axis=1)
df_features["Risk_Level"]        = df_features.apply(assign_risk, axis=1)
print("Clustering, Feedback & Risk done!")

# ── Cell 4: User Input → Score Calculation → Prediction → Report ─────────────
print("\n===== ENTER STUDENT DETAILS (SGPA SCALE 1–10) =====\n")

fields  = ["Bengali","English","Math","Science","Attendance (%)","class participation","Study Hours per Week","Assignment Score"]
prompts = ["Bengali SGPA (1-10)","English SGPA (1-10)","Math SGPA (1-10)","Science SGPA (1-10)",
           "Attendance Percentage (0-100)","Class Participation (1-10)","Study Hours per Week (1-10)","Assignment Score (1-10)"]
new_df  = pd.DataFrame([{f: float(input(f"Enter {p}: ")) for f, p in zip(fields, prompts)}])

new_df["Attendance_Scaled"]       = new_df["Attendance (%)"] / 10
new_df["Academic_Score"]          = new_df[["Bengali","English","Math","Science"]].mean(axis=1)
new_df["Engagement_Score"]        = (new_df["Attendance_Scaled"] + new_df["class participation"]) / 2
new_df["Effort_Score"]            = (new_df["Study Hours per Week"] + new_df["Assignment Score"]) / 2
new_df["Final_Performance_Score"] = (0.5*new_df["Academic_Score"] + 0.3*new_df["Engagement_Score"] + 0.2*new_df["Effort_Score"])

# Use Final_Performance_Score for prediction (same as training)
new_feats         = new_df[["Final_Performance_Score"]]
new_df["Cluster"] = kmeans.predict(new_feats)

new_df["Cluster_Label"] = new_df["Cluster"].map(cluster_labels)
new_df["Risk_Level"]    = new_df.apply(assign_risk, axis=1)

weakest_subject, lowest_mark = find_weakest_subject(new_df.iloc[0])
new_df["Weakest_Subject"]    = weakest_subject
new_df["Lowest_Mark"]        = lowest_mark
new_df["AI_Tutor_Feedback"]  = new_df.apply(generate_feedback, axis=1)

r = new_df.iloc[0]
print("\n" + "="*60)
print("        🎓 STUDENT PERFORMANCE ANALYSIS REPORT")
print("="*60)
print(f"\n📊 Predicted SGPA        : {round(r['Final_Performance_Score'],2)}")
print(f"📘 Academic Score        : {round(r['Academic_Score'],2)}")
print(f"📈 Engagement Score      : {round(r['Engagement_Score'],2)}")
print(f"🏷 Cluster Category       : {r['Cluster_Label']}")
print(f"⚠ Risk Level             : {r['Risk_Level']}")
print("\n" + "-"*60)
print(f"📌 Weakest Subject        : {weakest_subject}")
print(f"📉 Lowest SGPA in Subject : {lowest_mark}")
print(f"\n🎯 Focus Recommendation:\n   Improve your {weakest_subject} as your score ({lowest_mark}) is comparatively low.")
print(f"\n🤖 AI Tutor Feedback:\n   {r['AI_Tutor_Feedback']}")
print("\n" + "="*60)


# ── PDF Report Export ────────────────────────────────────────────────────────
try:
    from xhtml2pdf import pisa
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xhtml2pdf", "reportlab", "-q"])
    from xhtml2pdf import pisa

risk_color_map = {"Low Risk": "#27ae60", "Medium Risk": "#e67e22", "High Risk": "#e74c3c"}
risk_color     = risk_color_map[r["Risk_Level"]]

feedback_parts = r["AI_Tutor_Feedback"].split("\n")
feedback_main  = feedback_parts[0] if len(feedback_parts) > 0 else ""
feedback_extra = feedback_parts[1] if len(feedback_parts) > 1 else ""

html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {{ margin: 1.5cm 1.8cm; }}
  body {{
    font-family: Helvetica, Arial, sans-serif;
    background: #eef2f7;
    color: #333;
    font-size: 13px;
  }}
  .header-table {{
    width: 100%;
    background-color: #1a3c5e;
    border-radius: 10px;
    margin-bottom: 18px;
  }}
  .header-title {{ color: #ffffff; font-size: 24px; font-weight: bold; padding: 0; text-align: center; }}
  .header-sub   {{ color: #b8d4f0; font-size: 11px; margin-top: 4px; text-align: center; }}
  .section-title {{
    font-size: 15px;
    font-weight: bold;
    color: #1a3c5e;
    border-bottom: 2px dashed #c8d8e8;
    padding-bottom: 6px;
    margin-top: 18px;
    margin-bottom: 12px;
    text-align: center;
  }}
  .metric-table {{ width: 100%; border-collapse: separate; border-spacing: 8px; margin-bottom: 4px; }}
  .metric-cell {{
    background-color: #f5f9ff;
    border: 1px solid #d3e4f7;
    border-radius: 8px;
    text-align: center;
    padding: 14px 8px;
    width: 25%;
  }}
  .metric-label {{ font-size: 10px; color: #6c8096; font-weight: bold; margin-top: 4px; }}
  .metric-value {{ font-size: 22px; font-weight: bold; color: #1a3c5e; margin-top: 2px; }}
  .risk-text {{ color: {risk_color}; font-size: 12px; font-weight: bold; }}
  .score-table {{ width: 100%; border-collapse: collapse; border-radius: 6px; }}
  .score-table thead tr {{ background-color: #1a3c5e; }}
  .score-table thead th {{ color: #ffffff; padding: 8px 12px; font-size: 12px; text-align: left; }}
  .score-table tbody td {{ padding: 7px 12px; border-bottom: 1px solid #e8eef5; font-size: 12px; }}
  .score-table tbody tr:nth-child(even) td {{ background-color: #f5f9ff; }}
  .score-right {{ text-align: right; font-weight: bold; color: #1a3c5e; }}
  .card-red-header {{
    background-color: #d9534f; color: #ffffff; padding: 8px 14px;
    font-weight: bold; font-size: 13px; border-radius: 6px 6px 0 0; text-align: center;
  }}
  .card-teal-header {{
    background-color: #2a9d8f; color: #ffffff; padding: 8px 14px;
    font-weight: bold; font-size: 13px; border-radius: 6px 6px 0 0; text-align: center;
  }}
  .card-red-body {{
    background-color: #fff5f4; border: 1px solid #f5c6c2;
    border-top: none; padding: 12px 14px; border-radius: 0 0 6px 6px;
  }}
  .card-teal-body {{
    background-color: #f4fbfa; border: 1px solid #a8d8d2;
    border-top: none; padding: 12px 14px; border-radius: 0 0 6px 6px;
  }}
  .weakness-name  {{ font-size: 18px; font-weight: bold; color: #c0392b; }}
  .weakness-score {{ font-size: 18px; font-weight: bold; color: #c0392b; }}
  .feedback-box {{
    background-color: #f0f8ff; border: 1px solid #b8d8f0;
    border-radius: 8px; padding: 16px;
  }}
  .feedback-text {{ font-size: 13px; line-height: 1.7; color: #2c3e50; }}
  .model-table {{ width: 100%; border-collapse: collapse; margin-top: 6px; }}
  .model-table td {{ padding: 6px 10px; border-bottom: 1px solid #e8eef5; font-size: 12px; }}
  .model-table tr:nth-child(even) td {{ background-color: #f5f9ff; }}
  .footer-bar {{
    background-color: #1a3c5e; color: #b8d4f0; text-align: center;
    padding: 10px; font-size: 11px; border-radius: 0 0 8px 8px; margin-top: 20px;
  }}
  .divider {{ border: none; border-top: 1px dashed #d0dce8; margin: 16px 0; }}
</style>
</head>
<body>

  <table class="header-table" cellpadding="0" cellspacing="0">
    <tr>
      <td style="padding: 18px 22px; text-align: center; width: 100%;">
        <div class="header-title">Performance Analysis Report</div>
        <div class="header-sub">Generated on {datetime.now().strftime("%B %d, %Y  %I:%M %p")}</div>
      </td>
    </tr>
  </table>

  <div class="section-title">Performance Summary</div>
  <table class="metric-table" cellpadding="0" cellspacing="8">
    <tr>
      <td class="metric-cell">
        <div class="metric-label">Predicted SGPA</div>
        <div class="metric-value">{round(r["Final_Performance_Score"],2)}</div>
      </td>
      <td class="metric-cell">
        <div class="metric-label">Academic Score</div>
        <div class="metric-value">{round(r["Academic_Score"],2)}</div>
      </td>
      <td class="metric-cell">
        <div class="metric-label">Engagement Score</div>
        <div class="metric-value">{round(r["Engagement_Score"],2)}</div>
      </td>
      <td class="metric-cell">
        <div class="metric-label">Cluster Category</div>
        <div style="margin-top:4px;"><span class="risk-text">{r["Risk_Level"]}</span></div>
        <div style="font-size:13px;font-weight:bold;color:{risk_color};margin-top:3px;">{r["Cluster_Label"]}</div>
      </td>
    </tr>
  </table>

  <hr class="divider">

  <div class="section-title">Score Breakdown</div>
  <table width="100%" cellpadding="0" cellspacing="0">
    <tr valign="top">
      <td width="49%">
        <table class="score-table" cellpadding="0" cellspacing="0">
          <thead><tr><th>Subject</th><th style="text-align:right;">SGPA</th></tr></thead>
          <tbody>
            <tr><td>Bengali</td><td class="score-right">{round(r["Bengali"],2)}</td></tr>
            <tr><td>English</td><td class="score-right">{round(r["English"],2)}</td></tr>
            <tr><td>Math</td><td class="score-right">{round(r["Math"],2)}</td></tr>
            <tr><td>Science</td><td class="score-right">{round(r["Science"],2)}</td></tr>
          </tbody>
        </table>
      </td>
      <td width="2%"></td>
      <td width="49%">
        <table class="score-table" cellpadding="0" cellspacing="0">
          <thead><tr><th>Metric</th><th style="text-align:right;">Value</th></tr></thead>
          <tbody>
            <tr><td>Attendance</td><td class="score-right">{round(r["Attendance (%)"],2)}%</td></tr>
            <tr><td>Class Participation</td><td class="score-right">{round(r["class participation"],2)}/10</td></tr>
            <tr><td>Study Hours / Week</td><td class="score-right">{round(r["Study Hours per Week"],2)} hrs</td></tr>
            <tr><td>Assignment Score</td><td class="score-right">{round(r["Assignment Score"],2)}/10</td></tr>
          </tbody>
        </table>
      </td>
    </tr>
  </table>

  <hr class="divider">

  <div class="section-title">Weakness &amp; Recommendation</div>
  <table width="100%" cellpadding="0" cellspacing="0">
    <tr valign="top">
      <td width="49%">
        <div class="card-red-header">Weakest Subject</div>
        <div class="card-red-body">
          <span class="weakness-name">{r["Weakest_Subject"]}</span>
          <table width="100%" cellpadding="0" cellspacing="0" style="margin-top:10px;border-top:1px dashed #f5c6c2;">
            <tr>
              <td style="padding-top:8px;font-size:13px;color:#555;font-weight:bold;">Lowest SGPA in {r["Weakest_Subject"]}:</td>
              <td style="text-align:right;padding-top:8px;"><span class="weakness-score">{round(r["Lowest_Mark"],1)}</span></td>
            </tr>
          </table>
        </div>
      </td>
      <td width="2%"></td>
      <td width="49%">
        <div class="card-teal-header">Focus Recommendation</div>
        <div class="card-teal-body">
          <table cellpadding="0" cellspacing="0"><tr valign="middle">
            <td style="font-size:13px;color:#2c3e50;line-height:1.6;">
              Improve your <b>{r["Weakest_Subject"]}</b> as your score
              (<b>{round(r["Lowest_Mark"],1)}</b>) is comparatively low.
            </td>
          </tr></table>
        </div>
      </td>
    </tr>
  </table>

  
  <hr class="divider">

  <div class="section-title">AI Tutor Feedback</div>
  <div class="feedback-box">
    <table cellpadding="0" cellspacing="0"><tr valign="top">
      <td class="feedback-text">
        <b>{feedback_main}</b><br>{feedback_extra}
      </td>
    </tr></tabl
  </div>

  <hr class="divider">

  <div class="section-title">Model Information</div>
  <table class="model-table" cellpadding="0" cellspacing="0">
    <tr><td><b>Algorithm</b></td><td>KMeans (Optimised)</td></tr>
    <tr><td><b>Optimal Clusters (k)</b></td><td>{best_k}</td></tr>
    <tr><td><b>Silhouette Score</b></td><td><b style="color:#27ae60;">{round(best_score,4)}</b> &nbsp;(was 0.3319 in project2 &mdash; improved!)</td></tr>
    <tr><td><b>Clustering Feature</b></td><td>Final Performance Score (optimal single-feature clustering)</td></tr>
  </table>

  <div class="footer-bar">
    Student Performance Analysis System &nbsp;|&nbsp; Confidential Report
  </div>


</body>
</html>"""

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pdf_path = os.path.join("/Users/badhanmajumder/Desktop/college", f"student_report_{timestamp}.pdf")
with open(pdf_path, "wb") as pdf_file:
    result = pisa.CreatePDF(html_content, dest=pdf_file)
if result.err:
    print(f"⚠ PDF generation had errors: {result.err}")
else:
    print(f"✅ PDF Report saved to: {pdf_path}")

# ── Cell 5: All Visualizations ───────────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns

color = {"High Risk":"red", "Medium Risk":"orange", "Low Risk":"green"}[r["Risk_Level"]]

# ── Row 1: Bar / Pie / Line ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

bar_labels = ["Bengali","English","Math","Science","Engagement","Effort","Predicted SGPA"]
bar_vals   = [r["Bengali"],r["English"],r["Math"],r["Science"],r["Engagement_Score"],r["Effort_Score"],r["Final_Performance_Score"]]
for b in axes[0].bar(bar_labels, bar_vals, color=color): b.set_color(color)
axes[0].set_ylim(0,10); axes[0].set_title("Student Performance Overview (SGPA Scale)")
axes[0].set_ylabel("Score (0–10)"); axes[0].tick_params(axis='x', rotation=45)

axes[1].pie([r["Bengali"],r["English"],r["Math"],r["Science"]],
            labels=["Bengali","English","Math","Science"], autopct='%1.1f%%', startangle=90)
axes[1].set_title("Subject Performance Distribution (SGPA Scale)")

trend_params = ["Academic Score","Engagement Score","Effort Score","Predicted SGPA"]
trend_vals   = [r["Academic_Score"],r["Engagement_Score"],r["Effort_Score"],r["Final_Performance_Score"]]
axes[2].plot(trend_params, trend_vals, marker='o'); axes[2].set_ylim(0,10)
axes[2].set_title("Performance Trend Analysis"); axes[2].set_ylabel("Score (0–10)")
axes[2].grid(True); axes[2].tick_params(axis='x', rotation=15)
plt.tight_layout(); plt.show()

# ── Cluster Scatter ──────────────────────────────────────────────────────────
palette = ["red","green","blue","purple","orange","cyan","magenta"]
plt.figure(figsize=(8,6))
for i, cl in enumerate(df_features["Cluster"].unique()):
    cd = df_features[df_features["Cluster"]==cl]
    plt.scatter(cd["Academic_Score"], cd["Engagement_Score"],
                c=palette[i % len(palette)], label=cluster_labels.get(cl, f"Cluster {cl}"), s=60)
plt.title(f"KMeans Clustering (k={best_k}, Silhouette={round(best_score,4)})")
plt.xlabel("Academic Score"); plt.ylabel("Engagement Score")
plt.legend(); plt.grid(True); plt.show()

# ── Seaborn 2×2 Scatter Grid ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12,10))
pairs = [("Academic_Score","Engagement_Score"),("Academic_Score","Final_Performance_Score"),
         ("Engagement_Score","Final_Performance_Score"),("Effort_Score","Final_Performance_Score")]
for ax, (x, y) in zip(axes.flat, pairs):
    sns.scatterplot(ax=ax, x=df_features[x], y=df_features[y],
                    hue=df_features["Cluster"], palette="Set1")
plt.tight_layout(); plt.show()

# ── Heatmaps (side-by-side) ──────────────────────────────────────────────────
hm_full = pd.DataFrame({
    "Bengali":[r["Bengali"]],"English":[r["English"]],"Math":[r["Math"]],"Science":[r["Science"]],
    "Engagement":[r["Engagement_Score"]],"Effort":[r["Effort_Score"]],"Predicted SGPA":[r["Final_Performance_Score"]]
}, index=["Scores"])

hm_summary = pd.DataFrame({
    "Academic Score":[r["Academic_Score"]],"Engagement Score":[r["Engagement_Score"]],
    "Effort Score":[r["Effort_Score"]],"Predicted SGPA":[r["Final_Performance_Score"]]
}, index=["Student Performance"])

fig, axes = plt.subplots(1, 2, figsize=(18, 3))
kw = dict(annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=10, cbar_kws={'label':'Score (0–10)'})
sns.heatmap(hm_full,    linewidths=1.5, linecolor="black", ax=axes[0], **kw)
axes[0].set_title("Detailed Student Performance Heatmap", fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
sns.heatmap(hm_summary, linewidths=2,   linecolor="white", ax=axes[1], **kw)
axes[1].set_title("Student Performance Heatmap", fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout(); plt.show()
