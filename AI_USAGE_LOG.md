# AI Usage Log

## Scope and original submission

The team no longer has its original AI-usage records, as confirmed during this portfolio update. The tools, prompts, tasks, and reflections from the original March 2026 submission cannot be reconstructed accurately. This document does not claim to be a contemporaneous submission log.

The entry below records verifiable assistance during the post-assessment portfolio update on **November 18, 2026**.

## Tool and tasks

**Tool:** OpenAI Codex. The exact model/version was not recorded in the project artifacts.

**Requests:** Prepare the project for a GitHub portfolio; then implement the supplied assessment feedback. These are summaries of the user's requests, not invented historical prompts.

Codex assisted with:

- Inspecting files and both repository histories, recovering the later app source, preserving Git ancestry, and documenting contributor evidence.
- Correcting moving-average windows and sharing feature generation between training and live inference; preserving date alignment and the newest inference row.
- Simplifying redundant ETL error handling and adding actionable validation messages.
- Adding Streamlit charts, date/strategy controls, saved-result exploration, filters, and downloads; preventing API requests on ordinary control changes.
- Writing regression checks and running synthetic offline verification, without real credentials or claims about real model performance.
- Preparing the README, credits, feedback record, publication notes, and this log.

## Observed benefits and limits

The inspection found the same moving-average bug in two code paths. A shared implementation and numerical regression checks make that correction verifiable. UI tests confirmed that changing chart and strategy controls reuses loaded data instead of sending another API request.

The first repository URL contained an older source snapshot. The later collaborator repository supplied the missing app and additional contribution history. Clean Python 3.11 testing also exposed an OpenMP runtime requirement. AI assistance could not supply a real deployment URL, original AI-use records, or verified real-data results; those have not been invented.

## Lessons for team review

This session demonstrates why feature definitions must agree between training and inference, why a preprocessing fix requires retraining old models, and why synthetic tests do not establish predictive performance. It also distinguishes recorded Git contributions from a complete account of teamwork.

These are technical observations from the assistance session, not fabricated personal reflections from team members. The team should review and explain the changed code and add its own reflections before treating this as a complete course deliverable.

## Follow-up app build (November 18, 2026)

After the user confirmed that no deployment URL was available and requested a working app, Codex added a credential-free Streamlit demo. It uses deterministic synthetic prices, the shared feature pipeline, and compact in-memory XGBoost models, with explicit synthetic-data labels. Assistance also covered strategy fees, charts, visitor controls, deployment documentation, and regression checks. These outputs are demonstrations, not recovered original project results. Automated rendering and interaction checks passed; native browser visual review was incomplete because the preview integration failed and its fallback stalled. No public app was deployed.

## Publication and hosting follow-up

The user subsequently authorized the assistant to carry out publication and deployment. Assistance covered GitHub account/repository access checks, a Free-plan Render service configuration, a final source/credential review, and preserving both original development histories. A hosting connection is required before the app can be deployed; no public URL is invented.
