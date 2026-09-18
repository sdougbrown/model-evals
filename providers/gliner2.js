// GLiNER2 provider for the classifier eval — calls the local gliner2 service
// (hosts/rusty/serve-gliner.sh, port 8090) directly with the test vars,
// bypassing the chat prompt entirely: the encoder has no prompt, just
// finding + code + discussion assembled server-side.
//
// Returns the same JSON shape the deterministic assertions expect
// ({"classification": "...", "disposition": "..."}), so scores are directly
// comparable with the qwen/gemma runs on this eval.

const SERVICE_URL = process.env.GLINER2_URL || 'http://127.0.0.1:8090';

class Gliner2Provider {
  id() {
    return 'gliner2';
  }

  async callApi(_prompt, context) {
    const vars = context?.vars || {};
    const body = {
      finding_body: vars.finding_body || '',
      current_code: vars.current_code || '',
      discussion: Array.isArray(vars.discussion) ? vars.discussion : [],
      prior_classification: vars.prior_classification || null,
    };

    try {
      const res = await fetch(`${SERVICE_URL}/v1/classify-finding`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const detail = await res.text();
        return { error: `gliner2 service ${res.status}: ${detail.slice(0, 300)}` };
      }
      const result = await res.json();
      const output = JSON.stringify({
        classification: result.classification,
        disposition: result.disposition,
      });
      return {
        output,
        metadata: {
          classification_confidence: result.classification_confidence,
          disposition_confidence: result.disposition_confidence,
        },
      };
    } catch (err) {
      return { error: `gliner2 service unreachable: ${err.message}` };
    }
  }
}

module.exports = Gliner2Provider;
