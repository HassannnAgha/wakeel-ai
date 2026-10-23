// Checks that the inheritance calculator's number fields only take whole numbers within their limit.
// Run:  node tests/countfield_test.js   (pytest runs it through tests/test_ui.py)
const fs = require('fs');
const js = (() => { const src = fs.readFileSync(require('path').join(__dirname, '..', 'ui.py'), 'utf8'); const page = src.split('PAGE = r"""')[1].split('"""')[0]; return page.slice(page.indexOf('<script>') + 8, page.lastIndexOf('</script>')); })();
const grab = name => { const s = js.indexOf('function ' + name + '('); let d = 0;
  for (let j = js.indexOf('{', s); j < js.length; j++) { if (js[j] === '{') d++; else if (js[j] === '}') { d--; if (!d) return js.slice(s, j + 1); } } };
eval(grab('cleanCount'));
const assert = require('assert');
const clean = (typed, max) => { const input = {value: typed, dataset: {max: String(max)}}; cleanCount(input); return input.value; };
assert.strictEqual(clean('3', 4), '3');
assert.strictEqual(clean('5', 4), '4');          // more than four wives becomes four
assert.strictEqual(clean('9', 1), '1');          // a woman has at most one husband
assert.strictEqual(clean('1.5', 30), '15');      // no decimal point
assert.strictEqual(clean('2,5', 30), '25');
assert.strictEqual(clean('-3', 30), '3');        // no minus sign
assert.strictEqual(clean('1e3', 30), '13');      // no exponent
assert.strictEqual(clean('00001', 30), '1');     // no leading zeros
assert.strictEqual(clean('05', 30), '5');
assert.strictEqual(clean('0', 30), '0');
assert.strictEqual(clean('000', 30), '0');
assert.strictEqual(clean('', 30), '');
assert.strictEqual(clean('abc', 30), '');
assert.strictEqual(clean('99', 30), '30');
console.log('count fields ok');
