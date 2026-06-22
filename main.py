import os
import glob
import json
import time
import pandas as pd
from PIL import Image
from google import genai
from google.genai import types

# 1. HISTORY AGENT
class HistoryAgent:
    def __init__(self, history_dataframe):
        self.history_df = history_dataframe

    def analyze_user(self, user_id):
        user_record = self.history_df[self.history_df['user_id'] == user_id]
        if user_record.empty:
            return {"user_id": user_id, "risk_profile": "medium_risk", "risk_score": 0.5}

        row = user_record.iloc[0]
        past_claims = row['past_claim_count']
        manual_reviews = row['manual_review_claim']
        rejected = row['rejected_claim']
        recent_claims = row['last_90_days_claim_count']
        flags = str(row['history_flags']).strip().lower()

        rejection_rate = rejected / past_claims if past_claims > 0 else 0
        if flags != 'none' and flags != 'nan':
            return {"user_id": user_id, "risk_profile": "high_risk", "risk_score": 0.9}
        elif rejection_rate > 0.5 or manual_reviews > (past_claims * 0.5):
            return {"user_id": user_id, "risk_profile": "high_risk", "risk_score": 0.8}
        return {"user_id": user_id, "risk_profile": "low_risk", "risk_score": 0.1}

# 2. PRODUCTION EVIDENCE AGENT
class ProductionEvidenceAgent:
    def __init__(self, client, requirements_dataframe):
        self.client = client
        self.req_df = requirements_dataframe
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_images_processed = 0

    def _get_requirements_text(self, claim_object):
        relevant_reqs = self.req_df[(self.req_df['claim_object'] == claim_object) | (self.req_df['claim_object'] == 'all')]
        req_text = ""
        for idx, row in relevant_reqs.iterrows():
            req_text += f"- [{row['requirement_id']}]: {row['minimum_image_evidence']}\n"
        return req_text

    def evaluate_claim(self, claim_row, is_test=False):
        user_id = claim_row['user_id']
        conversation = claim_row['user_claim']
        claim_object = claim_row['claim_object']
        rules = self._get_requirements_text(claim_object)

        image_contents, image_ids = [], []
        paths = claim_row['image_paths'].split(';') if pd.notna(claim_row['image_paths']) else []

        for path in paths:
            clean_path = path.strip()
            if not clean_path: continue
            img_id = os.path.splitext(os.path.basename(clean_path))[0]
            image_ids.append(img_id)
            full_path = clean_path if clean_path.startswith("hackerrank-orchestrate-june26") else f"hackerrank-orchestrate-june26/dataset/{clean_path}"
            if os.path.exists(full_path):
                try:
                    img = Image.open(full_path)
                    image_contents.append(img)
                    self.total_images_processed += 1
                except: pass

        prompt = f"""You are an expert multi-modal fraud engine. Analyze this claim.
        Object Type: {claim_object}
        Conversation: {conversation}
        Guidelines: {rules}
        Respond strictly in JSON matching the submission schema requirements.
        """

        contents = [prompt] + image_contents
        try:
            self.total_calls += 1
            response = self.client.models.generate_content(
                model="gemini-2.5-flash", contents=contents,
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
            )
            if response.usage_metadata:
                self.total_input_tokens += response.usage_metadata.prompt_token_count
                self.total_output_tokens += response.usage_metadata.candidates_token_count
            return json.loads(response.text.strip().strip('`').replace('json', ''))
        except Exception as e:
            return {"evidence_standard_met": False, "claim_status": "not_enough_information", "issue_type": "unknown", "object_part": "unknown", "risk_flags": "none", "evidence_standard_met_reason": str(e), "claim_status_justification": "error", "supporting_image_ids": "none", "valid_image": False, "severity": "unknown"}

# 3. PRODUCTION ORCHESTRATOR
class ProductionOrchestrator:
    def __init__(self, history_agent, evidence_agent):
        self.history_agent = history_agent
        self.evidence_agent = evidence_agent

    def execute_pipeline(self, input_df, is_test=False):
        final_rows = []
        for idx, row in input_df.iterrows():
            history_meta = self.history_agent.analyze_user(row['user_id'])
            evidence_meta = self.evidence_agent.evaluate_claim(row, is_test=is_test)

            combined_risk = evidence_meta.get("risk_flags", "none")
            if history_meta.get("risk_profile") == "high_risk":
                combined_risk = "user_history_risk" if combined_risk == "none" else f"{combined_risk};user_history_risk"

            processed_record = {
                "user_id": row['user_id'], "image_paths": row['image_paths'], "user_claim": row['user_claim'], "claim_object": row['claim_object'],
                "evidence_standard_met": str(evidence_meta.get("evidence_standard_met", False)).lower(), "evidence_standard_met_reason": evidence_meta.get("evidence_standard_met_reason", ""),
                "risk_flags": combined_risk, "issue_type": evidence_meta.get("issue_type", "unknown"), "object_part": evidence_meta.get("object_part", "unknown"),
                "claim_status": evidence_meta.get("claim_status", "not_enough_information"), "claim_status_justification": evidence_meta.get("claim_status_justification", ""),
                "supporting_image_ids": evidence_meta.get("supporting_image_ids", "none"), "valid_image": str(evidence_meta.get("valid_image", False)).lower(), "severity": evidence_meta.get("severity", "unknown")
            }
            final_rows.append(processed_record)
            time.sleep(0.5)
        return pd.DataFrame(final_rows)

if __name__ == "__main__":
    print("Loading datasets...")
    # Initialize your API client and datasets here if running locally
    pass
