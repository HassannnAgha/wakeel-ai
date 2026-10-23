// Checks the word-by-word reveal's timing helpers in ui.py (the DOM part needs a browser).
// Run:  node tests/reveal_test.js   (pytest runs it through tests/test_ui.py)
const fs = require('fs');
const js = (() => { const src = fs.readFileSync(require('path').join(__dirname, '..', 'ui.py'), 'utf8'); const page = src.split('PAGE = r"""')[1].split('"""')[0]; return page.slice(page.indexOf('<script>') + 8, page.lastIndexOf('</script>')); })();
const grab = name => { const s = js.indexOf('function ' + name + '('); let d = 0;
  for (let j = js.indexOf('{', s); j < js.length; j++) { if (js[j] === '{') d++; else if (js[j] === '}') { d--; if (!d) return js.slice(s, j + 1); } } };
eval(['revealDuration', 'revealCount', 'snapWord'].map(grab).join('\n'));
const assert = require('assert');
assert.strictEqual(revealDuration(10), 1200);            // short answers still take a moment
assert.strictEqual(revealDuration(500), 3000);
assert.strictEqual(revealDuration(100000), 6000);        // long ones never take more than six seconds
assert.strictEqual(revealCount(100, 0, 1000), 0);
assert.strictEqual(revealCount(100, 500, 1000), 50);
assert.strictEqual(revealCount(100, 5000, 1000), 100);   // never past the end
const text = 'Bail was granted. [S1] done';
assert.strictEqual(snapWord(text, 0), 0);                // on a word start: nothing yet
assert.strictEqual(snapWord(text, 2), 4);                // mid-word: finish the word
assert.strictEqual(snapWord(text, 20), 22);              // a citation marker is not cut in half
assert.strictEqual(snapWord(text, 999), text.length);
console.log('reveal ok');
