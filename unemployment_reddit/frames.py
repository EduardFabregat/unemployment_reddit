"""Macro-frames: group the micro-topics into a handful of frames and add frame features.

``TOPIC_NAMES`` and ``FRAME_MAPPING`` describe one specific fitted model (the 48-topic
12_7_26 run). If you refit the topic model, re-derive them from the topic map.
"""

import numpy as np
import pandas as pd

from .config import N_MACRO_FRAMES, SEED

# Raw BERTopic name -> readable label used on the topic map.
TOPIC_NAMES = {
    "0_pua_for_the_you": "PUA Eligibility & Processing",
    "1_thank_thanks_you_this": "Community Support Expression",
    "2_call_number_calling_through": "Phone Line Congestion",
    "3_claim_my_to_it": "General Claim Maintenance",
    "4_mine_it_week_weeks": "Weekly Certification Tracking",
    "5_virginia_michigan_california_state": "Cross-State Portal Comparisons",
    "6_paid_payment_payments_it": "Payment Disbursement Status",
    "7_edd_to_the_and": "California EDD Logistics",
    "8_certify_certified_certification_it": "Benefit Certification Process",
    "9_600_300_extra_week": "Federal Stimulus Top-ups (FPUC)",
    "10_please_your_message_states": "Automated System Correspondence",
    "11_pelosi_trump_he_the": "Federal Legislative Politics",
    "12_id_verification_identity_verify": "Identity Verification Barriers",
    "13_people_wage_jobs_are": "Wage Stagnation & Job Realities",
    "14_same_it_update_issue": "System Status Mirroring",
    "15_peuc_eb_extension_weeks": "Federal Extension Programs",
    "16_benefits_claim_benefit_my": "Benefit Distribution Logistics",
    "17_quit_you_fired_employer": "Separation Legality (Quit/Fired)",
    "18_card_boa_bofa_debit": "BofA Debit Card Freezes",
    "19_file_filed_filing_week": "Initial Claim Filing Timeline",
    "20_tax_taxes_irs_refund": "IRS Tax Refunds",
    "21_lwa_payments_the_payment": "Lost Wages Assistance (LWA)",
    "22_covid_due_job_to": "Pandemic-Induced Separation Drivers",
    "23_appeal_hearing_the_appeals": "Appeals & Administrative Hearings",
    "24_deposit_direct_deposited_deposits": "Direct Deposit Configurations",
    "25_stimulus_the_check_checks": "Economic Impact Payments",
    "26_weekly_report_benefit_income": "Weekly Earnings Reporting",
    "27_applied_apply_approved_you": "Application Approval Status",
    "28_search_work_workshops_job": "Work Search Requirements",
    "29_unemployment_texas_you_job": "TWC Job Search Enforcement",
    "30_pending_still_processing_weeks": "Processing Delays & Pending Status",
    "31_overpayment_overpayments_waiver_the": "Overpayment Notices & Waivers",
    "32_extension_extensions_it_the": "Extension Implementation Gaps",
    "33_email_sent_emails_inbox": "Email Communications",
    "34_pennsylvania_pa_uc_claim": "Pennsylvania UC System",
    "35_1099_1099g_form_w2": "Tax Document Processing (1099-G)",
    "36_vec_they_the_is": "VEC System Infrastructure",
    "37_california_unemployment_ca_gov": "California EDD Web Infrastructure",
    "38_georgia_ga_gdol_dol": "Georgia GDOL Portal Logistics",
    "39_vaccine_covid_vaccinated_the": "Vaccine Mandates & Health Claims",
    "40_senator_office_governor_contact": "Legislative Escalation & Advocacy",
    "41_she_her_disability_work": "Disability Claims & Capacity",
    "42_wba_100_300_you": "Weekly Benefit Amount Calculations",
    "43_nj_jersey_unemployment_new": "New Jersey DOL Infrastructure",
    "44_california_claim_new_balance": "EDD Claim Balance Recalculations",
    "45_fraud_commit_committing_that": "Fraud Investigations & Locks",
    "46_gov2go_app_file_on": "Gov2Go Platform Portal Logistics",
    "47_job_jobs_work_overqualified": "Labor Underutilization & Mismatch",
}

# Frame column name -> the topic columns summed into it (read off the 6-cluster UMAP topic map).
FRAME_MAPPING = {
    # System bottlenecks, payment delays, support and administrative friction.
    "frame_bureaucratic_friction_and_delays": [
        "1_thank_thanks_you_this",
        "2_call_number_calling_through",
        "4_mine_it_week_weeks",
        "6_paid_payment_payments_it",
        "12_id_verification_identity_verify",
        "14_same_it_update_issue",
        "18_card_boa_bofa_debit",
        "21_lwa_payments_the_payment",
        "24_deposit_direct_deposited_deposits",
        "30_pending_still_processing_weeks",
        "31_overpayment_overpayments_waiver_the",
        "33_email_sent_emails_inbox",
        "45_fraud_commit_committing_that",
    ],
    # State portal navigation, extensions, eligibility and claim management.
    "frame_portal_logistics_and_claim_admin": [
        "0_pua_for_the_you",
        "3_claim_my_to_it",
        "5_virginia_michigan_california_state",
        "7_edd_to_the_and",
        "8_certify_certified_certification_it",
        "15_peuc_eb_extension_weeks",
        "16_benefits_claim_benefit_my",
        "19_file_filed_filing_week",
        "23_appeal_hearing_the_appeals",
        "27_applied_apply_approved_you",
        "32_extension_extensions_it_the",
        "34_pennsylvania_pa_uc_claim",
        "36_vec_they_the_is",
        "38_georgia_ga_gdol_dol",
        "40_senator_office_governor_contact",
        "44_california_claim_new_balance",
        "46_gov2go_app_file_on",
    ],
    # Job-search requirements, state infrastructure and disability/separation legality.
    "frame_work_search_and_compliance": [
        "10_please_your_message_states",
        "17_quit_you_fired_employer",
        "28_search_work_workshops_job",
        "29_unemployment_texas_you_job",
        "37_california_unemployment_ca_gov",
        "41_she_her_disability_work",
        "43_nj_jersey_unemployment_new",
        "47_job_jobs_work_overqualified",
    ],
    # Federal politics, macro labour-market conditions and pandemic policy.
    "frame_macro_policy_and_labor_realities": [
        "11_pelosi_trump_he_the",
        "13_people_wage_jobs_are",
        "22_covid_due_job_to",
        "25_stimulus_the_check_checks",
        "39_vaccine_covid_vaccinated_the",
    ],
    # Taxes, tax forms, earnings calculations and stimulus top-ups.
    "frame_taxation_and_financial_reporting": [
        "9_600_300_extra_week",
        "20_tax_taxes_irs_refund",
        "26_weekly_report_benefit_income",
        "35_1099_1099g_form_w2",
        "42_wba_100_300_you",
    ],
}


def add_frame_features(df: pd.DataFrame, mapping: dict = FRAME_MAPPING) -> tuple[pd.DataFrame, list[str]]:
    """Add one column per macro-frame (sum of its topic columns) to ``df`` in place.

    Returns ``(df, frame column names)``.
    """
    for frame, topic_cols in mapping.items():
        present = [c for c in topic_cols if c in df.columns]
        df[frame] = df[present].sum(axis=1)
    return df, list(mapping)


def topic_embedding_matrix(topic_model) -> np.ndarray:
    """Embedding of each real topic (the -1 outlier topic is dropped)."""
    return np.array(topic_model.topic_embeddings_)[1:]


def frame_count_diagnostics(X_topics: np.ndarray) -> pd.DataFrame:
    """Silhouette score and within-cluster variance (WCSS) for every candidate frame count K."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    rows = []
    for k in range(2, len(X_topics)):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_topics)
        wcss = sum(
            np.sum((X_topics[labels == i] - X_topics[labels == i].mean(axis=0)) ** 2)
            for i in range(k)
            if (labels == i).any()
        )
        rows.append({"k": k, "silhouette": silhouette_score(X_topics, labels), "wcss": wcss})
    return pd.DataFrame(rows)


def elbow_inflection(diagnostics: pd.DataFrame) -> int:
    """K at the sharpest bend of the WCSS curve (largest second difference)."""
    accel = np.diff(diagnostics["wcss"].to_numpy(), n=2)
    return int(diagnostics["k"].to_numpy()[1:-1][np.argmax(accel)])


def cluster_topics(X_topics: np.ndarray, n_frames: int = N_MACRO_FRAMES) -> np.ndarray:
    """Ward hierarchical clustering of topic embeddings into ``n_frames`` macro-frames."""
    from sklearn.cluster import AgglomerativeClustering

    return AgglomerativeClustering(n_clusters=n_frames, linkage="ward").fit_predict(X_topics)


def project_topics_2d(X_topics: np.ndarray) -> np.ndarray:
    """2-D UMAP projection of the topic embeddings for the topic map."""
    import umap

    n = len(X_topics)
    n_neighbors = int(np.clip(np.sqrt(n), 3, n - 1))
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        min_dist=0.1,
        metric="cosine",
        random_state=SEED,
    )
    return reducer.fit_transform(X_topics)


def topic_map_frame(topic_model, n_frames: int = N_MACRO_FRAMES) -> pd.DataFrame:
    """One row per topic with 2-D coordinates, macro-frame and readable name, ready to plot."""
    X_topics = topic_embedding_matrix(topic_model)
    coords = project_topics_2d(X_topics)
    info = topic_model.get_topic_info()
    raw_names = info.loc[info["Topic"] >= 0].sort_values("Topic")["Name"]
    return pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "cluster": cluster_topics(X_topics, n_frames),
            "topic_name": raw_names.map(TOPIC_NAMES).to_numpy(),
        }
    )
