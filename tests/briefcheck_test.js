// Checks the brief checker in ui.py on the sample petition: sections against the statute text, cases against the
// library, years against the calendar. Run:  node tests/briefcheck_test.js   (pytest runs it through tests/test_ui.py)
const fs = require('fs'), path = require('path');
const NL = String.fromCharCode(10);
const js = (() => { const src = fs.readFileSync(path.join(__dirname, '..', 'ui.py'), 'utf8'); const page = src.split('PAGE = r"""')[1].split('"""')[0]; return page.slice(page.indexOf('<script>') + 8, page.lastIndexOf('</script>')); })();
const grab = name => { const s = js.indexOf('function ' + name + '('); let d = 0;
  for (let j = js.indexOf('{', s); j < js.length; j++) { if (js[j] === '{') d++; else if (js[j] === '}') { d--; if (!d) return js.slice(s, j + 1); } } };
const line = name => { const s = js.indexOf('const ' + name + ' '); return js.slice(s, js.indexOf(NL, s)).replace('const ', 'var '); };
eval([line('PPC_RE'), line('CONST_RE'), line('LAW_RE'), line('CITE_RE'), line('COURTS'), line('SAMPLE_PETITION')].join(NL) + NL +
     ['lawPrefix', 'citeKey', 'checkBrief'].map(grab).join(NL));
const assert = require('assert');
const statutes = {'ppc:489-F': {title: 'Dishonestly issuing a cheque'}, 'crpc:497': {title: 'When bail may be taken in case of non-bailable offence'},
                  'const:10A': {title: 'Right to fair trial'}};
const cites = {'2013 SCMR 1205': {refs: {a: [], b: [], c: []}}, '2016 SC 808': {refs: {a: []}}};
const found = checkBrief(SAMPLE_PETITION, statutes, cites, 2026);
const by = ref => found.find(f => f.ref.startsWith(ref));
assert.strictEqual(by('Section 497').status, 'verified');
assert.strictEqual(by('Section 497').count, 2);                    // named twice, listed once
assert.strictEqual(by('Section 489-F').status, 'verified');
assert.strictEqual(by('Section 489-Z').status, 'wrong');           // no such section in the PPC
assert.strictEqual(by('Section 300').status, 'wrong');             // not in the CrPC we hold (repealed)
assert.strictEqual(by('Article 10A').status, 'verified');
assert.strictEqual(by('2013 SCMR 1205').status, 'verified');
assert.strictEqual(by('2013 SCMR 1205').judgments, 3);
assert.strictEqual(by('PLD 2016 SC 808').status, 'verified');      // the PLD form shares its key with "2016 SC 808"
assert.strictEqual(by('2019 SCMR 1234').status, 'check');          // not in the library: check it, not called fake
assert.strictEqual(by('PLD 2031 SC 77').status, 'wrong');          // a year that has not come
assert.deepStrictEqual(found.map(f => f.at), found.map(f => f.at).slice().sort((a, b) => a - b));
assert.strictEqual(checkBrief('No law here.', statutes, cites, 2026).length, 0);
console.log('brief check ok');
