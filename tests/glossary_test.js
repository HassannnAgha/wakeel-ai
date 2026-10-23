// Checks the legal-word glossary in ui.py: which words are marked, and that a longer form wins over a shorter one.
// Run:  node tests/glossary_test.js   (pytest runs it through tests/test_ui.py)
const fs = require('fs'), path = require('path');
const NL = String.fromCharCode(10);
const js = (() => { const src = fs.readFileSync(path.join(__dirname, '..', 'ui.py'), 'utf8'); const page = src.split('PAGE = r"""')[1].split('"""')[0]; return page.slice(page.indexOf('<script>') + 8, page.lastIndexOf('</script>')); })();
const grab = name => { const s = js.indexOf('function ' + name + '('); let d = 0;
  for (let j = js.indexOf('{', s); j < js.length; j++) { if (js[j] === '{') d++; else if (js[j] === '}') { d--; if (!d) return js.slice(s, j + 1); } } };
const block = js.slice(js.indexOf('const GLOSSARY = ['), js.indexOf('let GLOSS_RES = null;'));
eval(block.replace('const GLOSSARY', 'var GLOSSARY') + NL + 'var GLOSS_RES = null;' + NL + ['glossRes', 'glossFind', 'glossMatches'].map(grab).join(NL));
const assert = require('assert');
const ids = t => glossMatches(t).map(m => m.id);
assert.deepStrictEqual(ids('The FIR was lodged first; the challan followed.'), ['fir', 'challan']);
assert.deepStrictEqual(ids('the first information'), []);                          // "first" is not FIR
assert.deepStrictEqual(ids('a non-cognizable offence, then a cognizable offence'), ['noncog', 'cog']);
assert.deepStrictEqual(ids('a non-bailable offence and a bailable offence'), ['nonbail', 'bailable']);
assert.deepStrictEqual(ids('pre-arrest bail and then post-arrest bail'), ['prearrest', 'postarrest']);
assert.deepStrictEqual(ids('outside the prohibitory clause of section 497'), ['prohibitory']);
assert.deepStrictEqual(ids('a superdari application; physical remand'), ['superdari', 'remand']);
assert.deepStrictEqual(ids('FIR, FIR and FIR'), ['fir']);                          // only the first time
assert.deepStrictEqual(ids('remanded and decreed'), []);                           // whole words only
const ur = String.fromCharCode(0x627, 0x6cc, 0x641) + ' ' + String.fromCharCode(0x622, 0x626, 0x6cc) + ' ' + String.fromCharCode(0x622, 0x631);
assert.deepStrictEqual(ids('میری ' + ur + ' درج'), ['fir']);   // Urdu: "FIR" written out
assert.ok(GLOSSARY.every(g => g.en && g.ur && (g.ci || g.cs || g.urp)));
assert.strictEqual(new Set(GLOSSARY.map(g => g.id)).size, GLOSSARY.length);
console.log('glossary ok');
