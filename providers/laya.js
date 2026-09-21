// Laya provider for the classifier eval — zero-shot System-1 typed decisions.
// Calls the local gliner2 service's /v1/laya/classify-finding endpoint (the
// service hosts both engines). Returns the same JSON shape as the GLiNER
// provider so the deterministic assertions are directly comparable.

const SERVICE_URL = process.env.GLINER2_URL || 'http://127.0.0.1:8090';

class LayaProvider {
  id() {
    return 'laya';
  }

  async callApi(_prompt, context) {
    const vars = context?.vars || {};
    const body = {
      finding_body: vars.finding_body || '',
      current_code: vars.current_code || '',
      discussion: Array.isArray(vars.discussion) ? vars.discussion : [],
      prior_classification: vars.prior_classification || null,
      model: process.env.LAYA_VARIANT || 'base',
    };

    try {
      const res = await fetch(`${SERVICE_URL}/v1/laya/classify-finding`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const detail = await res.text();
        return { error: `laya service ${res.status}: ${detail.slice(0, 300)}` };
      }
      const result = await res.json();
      const output = JSON.stringify({
        classification: result.classification,
        disposition: result.disposition,
      });
      return {
        output,
        metadata: {
          classification_probabilities: result.classification_probabilities,
          disposition_probabilities: result.disposition_probabilities,
          classification_confidence: result.classification_confidence,
          disposition_confidence: result.disposition_confidence,
        },
      };
    } catch (err) {
      return { error: `laya service unreachable: ${err.message}` };
    }
  }
}

module.exports = LayaProvider;
