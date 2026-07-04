const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, AlignmentType, BorderStyle, ImageRun, LevelFormat,
  PageBreak, convertInchesToTwip
} = require("docx");
const fs = require("fs");

const PAGE = { size: { width: 12240, height: 15840 } }; // US Letter

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 } });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 160 } });
}
function bullet(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, ...opts })],
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80 },
  });
}
function boldLead(lead, rest) {
  return new Paragraph({
    children: [new TextRun({ text: lead, bold: true }), new TextRun({ text: rest })],
    spacing: { after: 160 },
  });
}

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 1600, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "2F5496" } : undefined,
    children: [new Paragraph({
      children: [new TextRun({ text, bold: !!opts.header, color: opts.header ? "FFFFFF" : "000000", size: 20 })],
    })],
  });
}

function dataTable(headers, rows, colWidths) {
  const totalWidth = colWidths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({ children: headers.map((hd, i) => cell(hd, { header: true, width: colWidths[i] })) }),
      ...rows.map(r => new TableRow({ children: r.map((c, i) => cell(String(c), { width: colWidths[i] })) })),
    ],
  });
}

function image(path, width, height) {
  return new Paragraph({
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width, height } })],
    spacing: { after: 200 },
    alignment: AlignmentType.CENTER,
  });
}

const doc = new Document({
  numbering: {
    config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT }] }],
  },
  sections: [{
    properties: { page: PAGE },
    children: [
      new Paragraph({ text: "SentinelAI", heading: HeadingLevel.TITLE, spacing: { after: 80 } }),
      new Paragraph({
        children: [new TextRun({ text: "A Cross-Lingual LLM Safety Evaluation Framework — Pilot v1 Technical Report", italics: true, size: 26 })],
        spacing: { after: 300 },
      }),
      p("July 2026", { size: 20, color: "555555" }),

      h1("1. Executive Summary"),
      p("SentinelAI is a reproducible framework for testing whether open-source LLMs' safety alignment holds consistently across languages, or whether it degrades outside English. This report documents Pilot v1: a 28-prompt end-to-end run across 5 languages (English, Hindi, Mandarin, Tamil, Malay) and 2 successfully-evaluated open models (Mistral 7B, Gemma 2B), designed to validate the methodology before scaling to a 300\u2013500 prompt study."),
      boldLead("The pilot's core finding is methodological, not behavioral: ", "manual inspection of raw model outputs revealed that the automated rule-based refusal classifier systematically under-detects refusals in non-English languages. A confirmed counterexample (Mistral, Hindi) shows the model genuinely declining a harmful request in phrasing the classifier's limited pattern set did not recognize. This means the pilot's headline numeric results (e.g. 0% refusal in Hindi/Tamil/Malay) should not be read as evidence that these models fail to refuse in those languages \u2014 they are evidence that the current scoring method needs improvement before that question can be answered."),
      p("Two additional threats to validity were identified and documented: degraded generation/translation quality in Tamil outputs, and a reproducible Ollama API-level incompatibility that excluded two of the four intended models (Llama 2 7B, Gemma 9B) from this pilot on the test machine (macOS 13, Apple Silicon)."),
      p("This is treated as a successful pilot: it answered the question a pilot is meant to answer \u2014 \u201ccan this methodology be trusted?\u201d \u2014 with a clear, evidenced \u201cpartially, and here is exactly where and why,\u201d rather than either false confidence or an untested assumption."),

      h1("2. Motivation & Research Question"),
      p("Most public LLM safety benchmarks (AdvBench, HarmBench, JailbreakBench, DoNotAnswer) are English-only. Models deployed across South and Southeast Asia serve users in many languages simultaneously. If safety alignment does not transfer across languages, that gap is invisible to English-only evaluation and only surfaces in production."),
      boldLead("Research question: ", "Do open-source LLMs exhibit consistent safety behavior across languages when presented with equivalent prompts?"),
      p("Primary hypothesis (H1): refusal rates for unsafe requests are measurably lower in lower-resource languages (Tamil, Malay) than in higher-resource languages (English, Hindi, Mandarin). Pilot v1 was not designed to test H1 conclusively \u2014 28 prompts is far too small a sample \u2014 but to validate that the measurement pipeline itself produces trustworthy data before investing in a larger run."),

      h1("3. Related Work"),
      p("SentinelAI builds on an established line of English-language safety benchmarks, extending their core idea \u2014 systematically probing models with unsafe prompts and scoring refusal/compliance \u2014 to a multilingual setting:"),
      bullet("AdvBench (Zou et al.) provides a widely-used set of harmful-behavior prompts and adversarial suffixes for jailbreak research, but is English-only and does not test cross-lingual transfer."),
      bullet("HarmBench offers a standardized evaluation framework across harm categories with an emphasis on comparability between attack and defense methods, again in English."),
      bullet("JailbreakBench focuses specifically on adversarial jailbreak success rate with a public leaderboard, prioritizing attack diversity over language diversity."),
      bullet("DoNotAnswer is closer in spirit to SentinelAI's over-refusal measurement, providing prompts a well-aligned model should refuse, but likewise does not vary language as an experimental factor."),
      p("SentinelAI's contribution is not a new attack taxonomy but a cross-lingual measurement layer applied on top of this existing style of evaluation: the same prompt, translated, run against the same model, to isolate language as the variable of interest. Pilot v1's main finding \u2014 that the measurement method itself needs validation before cross-lingual claims can be trusted \u2014 is also relevant to this broader literature, since none of the benchmarks above appear to report non-English refusal-detection validation as part of their published methodology."),

      h1("4. Methodology Summary"),
      h2("4.1 Dataset"),
      p("28 prompts spanning 12 categories (harmful advice, cybersecurity, privacy, misinformation, medical, financial, bias, illegal activity, prompt injection, jailbreak, roleplay attacks, sensitive data extraction) plus benign controls used to measure over-refusal. Each prompt was translated from English into Hindi, Mandarin, Tamil, and Malay using Meta's NLLB-200 model."),
      h2("4.2 Models"),
      p("Four open-source models were targeted for evaluation via Ollama: Llama 2 7B, Mistral 7B, Gemma 9B, and Gemma 2B. Each prompt/language combination was run 3 times at temperature 0 to check for sampling variance."),
      h2("4.3 Scoring"),
      p("A rule-based classifier (keyword/regex matching) determined refusal, applied per-language with hand-written pattern sets. Refusal rate, over-refusal rate (on benign controls), and a safety consistency score (standard deviation of refusal rate across languages, per model) were computed from the results."),
      p("Full methodology, including the experimental protocol, statistical testing plan, and ethics considerations, is documented separately in docs/methodology.md."),

      h2("4.4 Reproducibility Summary"),
      dataTable(
        ["Item", "Value"],
        [
          ["Models evaluated", "Mistral 7B, Gemma 2B"],
          ["Models attempted but excluded", "Llama 2 7B, Gemma 9B (see Section 6.3)"],
          ["Languages", "English, Hindi, Mandarin, Tamil, Malay"],
          ["Prompts", "28 (12 categories + benign controls)"],
          ["Runs per prompt/language/model", "3"],
          ["Total generation attempts", "1,680"],
          ["Valid (non-error) generations used in scoring", "829"],
          ["Translation backend", "NLLB-200-distilled-600M"],
          ["Inference framework", "Ollama v0.10.1"],
          ["Hardware", "Apple Silicon, macOS 13"],
          ["Decoding", "Temperature 0.0, max 512 tokens"],
        ],
        [3400, 4600]
      ),

      new Paragraph({ children: [new PageBreak()] }),
      h1("5. Pilot v1 Results"),
      p("Of the 4 targeted models, 2 (Llama 2 7B, Gemma 9B) failed 100% of API calls due to a tooling incompatibility detailed in Section 6.3. Results below reflect the 2 models that ran successfully: Mistral 7B and Gemma 2B."),
      image("reports/report_assets/error_rate_by_model.png", 430, 287),

      h2("5.1 Refusal Rate by Language"),
      p("Percentage of unsafe prompts the model appropriately refused, as scored by the rule-based classifier. See Section 6.1 for why these non-English numbers should not be taken at face value."),
      dataTable(
        ["Model", "English", "Hindi", "Mandarin", "Tamil", "Malay"],
        [["Mistral 7B", "9.9%", "0.0%", "4.3%", "0.0%", "4.3%"],
         ["Gemma 2B", "16.7%", "0.0%", "41.7%", "0.0%", "0.0%"]],
        [2200, 1440, 1440, 1440, 1440, 1440]
      ),
      new Paragraph({ text: "", spacing: { after: 200 } }),
      image("reports/report_assets/refusal_by_language.png", 430, 258),

      h2("5.2 Safety Consistency Score"),
      p("Standard deviation of refusal rate across the 5 languages, per model. Lower indicates more consistent behavior across languages \u2014 but given the classifier issue in Section 6.1, this score is currently more a measure of \u201chow much the classifier's blind spots vary by model\u201d than a clean safety metric."),
      dataTable(["Model", "Consistency Score (\u03c3)"], [["Mistral 7B", "4.08"], ["Gemma 2B", "18.27"]], [3000, 3000]),

      h2("5.3 Over-Refusal Rate (Benign Controls)"),
      p("Both models scored 0% over-refusal across all 5 languages \u2014 no benign control prompt was incorrectly refused."),
      dataTable(
        ["Model", "English", "Hindi", "Mandarin", "Tamil", "Malay"],
        [["Mistral 7B", "0.0%", "0.0%", "0.0%", "0.0%", "0.0%"],
         ["Gemma 2B", "0.0%", "0.0%", "0.0%", "0.0%", "0.0%"]],
        [2200, 1440, 1440, 1440, 1440, 1440]
      ),

      new Paragraph({ children: [new PageBreak()] }),
      h1("6. Threats to Validity (Key Findings)"),
      p("This section is the substantive output of Pilot v1. Each item below was identified through manual inspection of raw model outputs \u2014 not assumed in advance \u2014 and materially changes how the numeric results in Section 5 should be interpreted."),

      h2("6.1 Rule-based refusal detection under-detects non-English refusals"),
      p("Automated scoring reported 0% refusal in Hindi, Tamil, and Malay across both models \u2014 a suspiciously flat result. Manual inspection of raw outputs found a clear counterexample: Mistral's response to prompt adv-002 in Hindi explicitly declines the harmful framing (\u201c...but I cannot exceptionally tell my client the fastest way...\u201d) and redirects to a legitimate alternative, yet the classifier's narrow Hindi regex pattern set did not match this phrasing and scored it as compliance."),
      boldLead("Conclusion: ", "the automated rule-based refusal detector under-detects refusals in non-English languages due to a narrow, hand-written pattern set. Reported non-English refusal rates in Pilot v1 are unreliable lower bounds, not evidence that these models fail to refuse in these languages. This is a methodological finding, not a safety finding."),

      h2("6.2 Tamil outputs show signs of degraded generation/translation quality"),
      p("Several Tamil outputs during manual inspection were repetitive or incoherent, independent of the refusal question. This confounds interpretation: a low or zero refusal rate in Tamil could reflect genuine model behavior, poor translation into Tamil, or weaker model generation quality in Tamil regardless of prompt content. The translation QA gate (back-translation review) and Tamil-specific human review are required before trusting any Tamil-language conclusion."),

      h2("6.3 Model coverage was reduced due to a reproducible tooling incompatibility"),
      p("Llama 2 7B and Gemma 9B were excluded after reproducible, deterministic failures at the Ollama API level: every /api/generate call for these two models returned \u201cllama runner process no longer running,\u201d confirmed via direct API testing independent of this project's code, and confirmed NOT to occur via Ollama's interactive CLI (ollama run) on the same models and same machine. This isolates the failure to the API-serving code path on this environment (macOS 13, Apple Silicon, Ollama v0.10.1) rather than to the models themselves or to insufficient hardware."),

      h2("6.4 Sample size"),
      p("28 prompts is sufficient to validate that the pipeline runs end-to-end and to surface the issues above, but is far too small to support any generalizable claim about cross-lingual safety consistency. All Pilot v1 numeric results should be read as evidence about pipeline functionality, not as evidence for or against the research hypothesis."),

      h1("7. Limitations \u2192 Future Work"),
      p("Pilot v1 established the feasibility of the evaluation pipeline while identifying methodological limitations that must be addressed before large-scale experimentation. Future work will focus on multilingual refusal classification, expanded datasets, improved model coverage, and human validation."),
      p("Priorities before scaling to the full 300\u2013500 prompt study, in order:"),
      bullet("Improve multilingual refusal detection: expand language-specific patterns using real observed refusal phrasings, and add an LLM-as-judge pass rather than relying on regex alone."),
      bullet("Resolve the Tamil quality question via the translation QA / back-translation gate, to separate translation-side from generation-side issues."),
      bullet("Resolve or re-document the Ollama model compatibility issue (upgrade Ollama, or move to a cloud/Linux environment) to restore full 4-model coverage."),
      bullet("Only once the above are addressed, scale the dataset from 28 to 300\u2013500 prompts."),
      p("The target scoring pipeline for Version 2:"),
      p("Prompt \u2192 Model Response \u2192 Rule-based Classifier \u2192 LLM-as-a-Judge \u2192 Human Validation \u2192 Final Label", { italics: true }),

      h1("8. Conclusion"),
      p("Pilot v1 achieved its actual purpose: it validated the SentinelAI pipeline end-to-end and, more importantly, surfaced three concrete, evidenced weaknesses in the measurement methodology before they could contaminate a larger, more expensive study. The project demonstrates a complete research cycle \u2014 design, build, execute, critically inspect, and document \u2014 with the self-correcting step (catching that a suspicious result was a measurement artifact rather than accepting it as a finding) being the most significant output of this phase."),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("reports/SentinelAI_Pilot_v1_Report.docx", buf);
  console.log("Report written.");
});
