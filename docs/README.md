# Documentation guide

Read product plans as intended direction, not proof that a feature has shipped.
Some older status sections describe earlier revisions. Current code, tests, and
dated delivery notes take precedence for implementation status.

| Document | What it governs |
|---|---|
| [Plan risk review](PLAN_RISK_REVIEW.md) | Frontend/backend findings, launch blockers, Indian regulatory boundaries and effective-date caveats |
| [Business implementation plan](BUSINESS_IMPLEMENTATION_PLAN.md) | Consolidated launch scope, business model, pricing hypotheses, delivery backlog, and release gates |
| [Implementation context](IMPLEMENTATION_CONTEXT.md) | Engine inventory, conventions, tests, and implementation notes |
| [Competitor feature delivery](COMPETITOR_FEATURE_DELIVERY.md) | September 2026 additions, current limitations, and next work |
| [Product blueprint](PRODUCT_BLUEPRINT.md) | Analytics positioning, five-section navigation, and long-term product design |
| [Competitor gap analysis](COMPETITOR_GAP_ANALYSIS.md) | Historical comparison and adoption priorities |
| [App workflow and wireframe](APP_WORKFLOW_AND_WIREFRAME.md) | User journeys, product layers, and planned web/mobile flows |
| [Compliant feature wireframes](COMPLIANT_FEATURE_WIREFRAMES.md) | Descriptive terminology and launch-facing UX requirements |
| [Business model](BUSINESS_MODEL_NO_SEBI_ADVISOR.md) | Intended analytics-only business scope and subscription planning |
| [Production architecture](PRODUCTION_ARCHITECTURE.md) | Future API, identity, services, streaming, and data infrastructure |
| [Fundraising roadmap](FUNDRAISING_ROADMAP_INDIA.md) | Business preparation, outreach planning, and milestones |

Implementation principles retained from these documents:

- Users choose the universe and conditions; results describe observed data.
- Preserve local SQLite workflows while keeping new UI separate from analysis code.
- Keep data timestamps and missing coverage visible.
- Retain lazy page routing under the existing primary sections.
- Treat cloud services, licensed feeds, mobile delivery, and broker integrations as
  separate projects with explicit dependencies.
- Business and regulatory statements in planning documents are project assumptions;
  this implementation does not establish their legal correctness.
