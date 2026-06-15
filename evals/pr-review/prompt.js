const fs = require('fs');
const path = require('path');

module.exports = async function ({ vars }) {
  const diff = fs.readFileSync(
    path.join(__dirname, 'testdata', vars.diff_file),
    'utf8'
  // Escape Nunjucks delimiters — diff content may contain Go/Jinja template
  // syntax like {{.Field}} that promptfoo re-renders through Nunjucks.
  ).replace(/\{\{/g, '{ {').replace(/\}\}/g, '} }');

  return [
    {
      role: 'user',
      content: `Review the following pull request diff from the ${vars.repo} repository.

Focus on:
- Correctness: logic errors, off-by-ones, race conditions, missing edge cases
- Safety: unhandled errors, nil/null dereferences, resource leaks
- Quality: unnecessary complexity, missing tests for new behaviour
- Actionability: every issue you flag should include a concrete suggestion

Be specific — cite line numbers or function names. Do not invent issues that aren't visible in the diff. If you have no concerns, say so clearly rather than padding the review.

\`\`\`diff
${diff}
\`\`\``,
    },
  ];
};
