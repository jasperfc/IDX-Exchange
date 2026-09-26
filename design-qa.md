# Design QA

- Source visual truth: https://california-home-intelligence.streamlit.app/
- Implementation: http://127.0.0.1:8501/
- Source state: Home and Estimate pages inspected in the Codex in-app browser.
- Implementation state: Home, Estimate, populated estimate result, Model Insights, and Analyst Tools.
- Comparison intent: Adapt the source site's editorial hierarchy and website-like information architecture to the existing green California AVM brand; this is not a pixel-for-pixel clone.
- Viewport: 615 × 863 CSS px for the primary implementation captures; device scale factor 1.
- Evidence: Browser-rendered screenshots for the reference Home/Estimate views and implementation Home/Estimate/result views were opened and visually inspected during the current QA run. The in-app browser capture API did not expose persistent local screenshot paths.

## Required fidelity surfaces

- Fonts and typography: Passed. Georgia display headings and sans-serif UI copy create the intended editorial hierarchy; mobile wrapping remains readable.
- Spacing and layout rhythm: Passed. Hero, evidence strip, feature cards, form, and result card use consistent section spacing and 14–16px radii. The mobile evidence grid reflows to two columns.
- Colors and visual tokens: Passed. Forest green, warm paper, sage highlight, muted copy, and restrained borders are consistent across all tested pages. No decorative gradients remain.
- Image quality and asset fidelity: Passed for the chosen adaptation. The optimized California residential hero photograph is used as a responsive background; no placeholder assets are present.
- Copy and content: Passed. Product value, evidence, model limits, analyst scope, and appraisal disclaimer are visible in plain language.

## Full-view comparison evidence

- The implementation now matches the reference's core structural qualities: product header, horizontal navigation, editorial hero, evidence-led summary, use-case cards, focused page introductions, and a trust/method section.
- The implementation intentionally preserves the user's forest-green identity and existing prediction workflow instead of copying the reference palette or market-map features.

## Focused-region evidence

- Header/navigation: Active-page state is visible and keyboard-accessible through Streamlit radio controls. Mobile spacing was tightened after the first capture.
- Estimate form: Eight essential fields remain visible; technical and optional fields are grouped under Additional details.
- Result state: Estimate, reference range, model agreement, plain-language guidance, and collapsed technical details were all rendered successfully with default inputs.

## Interaction verification

- Home CTA navigates to Estimate.
- All four top-navigation destinations render.
- Additional details expander is available.
- Default estimate submission completes and renders the result summary.
- Model comparison and encoded-feature details remain available as expanders.
- Analyst CSV template/upload controls render.
- No application error was visible in the tested flow.

## Comparison history

### Iteration 1

- P2: Brand-name contrast appeared too faint in the narrow capture.
- Fix: Applied explicit high-contrast brand-name and subtitle colors.
- P2: Navigation was crowded at the narrow breakpoint.
- Fix: Reduced mobile navigation padding/type size and enforced non-wrapping labels with horizontal overflow.

### Iteration 2

- Post-fix evidence: Home, Estimate, estimate result, Model Insights, and Analyst Tools rendered without broken layout or blocked controls.

### Iteration 3

- P1: The Home CTA attempted to mutate the `page` widget state after widget instantiation and raised a StreamlitAPIException.
- Fix: Moved navigation into a widget callback, which runs before the next render. The CTA was clicked again and successfully opened Estimate.
- P2: Streamlit's fixed application header overlaid the custom navigation as a pale strip.
- Fix: Removed the framework header from layout and rechecked the Home screen; the brand and navigation now render unobstructed.
- P2: Analyst Tools lacked the promised guided workflow.
- Fix: Added schema and encoded-example downloads, a separate validation/upload step, validation metrics, and a review/export step.
- P2: The source-informed LightGBM explanation was shortened too aggressively.
- Fix: Added the full plain-language explanation above the held-out metrics.
- P2: The text-only hero lacked a visual focal point and the heading dominated narrower views.
- Fix: Added a project-specific California residential hero image and reduced hero type and vertical proportions at desktop and mobile breakpoints.
- Post-fix evidence: CTA navigation, unobstructed header, Analyst Tools steps, and exact LightGBM explanation were browser-verified.
- Remaining P0/P1/P2 findings: None.

### Iteration 4

- P2: The navigation still used visible radio indicators and looked like a form control rather than website navigation.
- Fix: Replaced the radio widget with a segmented control and restyled it as text tabs with a forest-green active underline. At narrow widths the four destinations reflow to a balanced two-by-two tab grid.
- Post-fix evidence: All four tabs render without circular indicators, and the Home CTA still changes the selected tab to Estimate without an exception.
- Remaining P0/P1/P2 findings: None.

### Iteration 5

- P2: The hero photograph read as a separate content block rather than part of the Home identity.
- Fix: Converted the optimized photograph into the Home hero background, with an inset translucent warm-paper panel for copy contrast and a responsive crop. Shortened the headline to prevent awkward narrow-width word breaks.
- Post-fix evidence: The background image, text panel, CTA, and two-by-two narrow navigation were rendered together and remained readable.
- Remaining P0/P1/P2 findings: None.

### Iteration 6

- P2: Model Insights reported only aggregate scores, making model differentiation and high-risk segments difficult to understand.
- Fix: Added error-distribution metrics and charts, P90/P95 tail comparison, price-quintile MdAPE trends, LightGBM residual bias, test-month band boundaries, and decision-oriented takeaways.
- Validation: Recomputed all summaries from the saved five models and the 12,442-row June 2026 encoded test set, then matched the aggregate values against `metrics_summary.csv` and Notebook 6.
- Post-fix evidence: Model Insights rendered all new charts and analysis without an application error at the narrow responsive viewport.
- Remaining P0/P1/P2 findings: None.

## Follow-up polish

- P3: A wider desktop-specific QA pass could further tune the maximum hero width and navigation spacing.

final result: passed
