// Checks the statute linker in ui.py against citation formats taken from the judgments.
// Run:  node tests/linker_test.js   (pytest runs it through tests/test_ui.py)
const fs = require('fs');
const js = (() => { const src = fs.readFileSync(require('path').join(__dirname, '..', 'ui.py'), 'utf8'); const page = src.split('PAGE = r"""')[1].split('"""')[0]; return page.slice(page.indexOf('<script>') + 8, page.lastIndexOf('</script>')); })();
const grab = name => { const s = js.indexOf('function ' + name + '('); let d = 0;
  for (let j = js.indexOf('{', s); j < js.length; j++) { if (js[j] === '{') d++; else if (js[j] === '}') { d--; if (!d) return js.slice(s, j + 1); } } };
const line = name => js.slice(js.indexOf('const ' + name + ' ='), js.indexOf('\n', js.indexOf('const ' + name + ' =')));
var CITES = {};
var STATUTES = {'ppc:489-F': {}, 'const:199': {}, 'crpc:497': {}, 'crpc:561A': {}, 'crpc:154': {}, 'crpc:249A': {},
                'cpc:12': {}, 'cpc:115': {}, 'gwa:25': {}, 'fca:5': {}, 'mflo:7': {}, 'dmma:2': {}};
eval((['PPC_RE', 'CONST_RE', 'LAW_RE'].map(line).join('\n') + '\n' + js.slice(js.indexOf('const LAWS ='), js.indexOf('function statuteLaw')) +
      ['lawPrefix', 'linkStatutes', 'statuteLaw'].map(grab).join('\n')).split('const ').join('var '));
const keys = s => [...linkStatutes(s).matchAll(/data-stat="([^"]+)"/g)].map(m => m[1]);
const cases = [
  ['bail under section 497(5) Cr.P.C. was refused', ['crpc:497']],
  ['petition u/s 561-A Cr.P.C, accepted', ['crpc:561A']],
  ['Section 561A CrPC', ['crpc:561A']],
  ['section 154, Cr.P.C. requires', ['crpc:154']],
  ['Section 249-A Cr.P.C', ['crpc:249A']],
  ['section 497 of the Code of Criminal Procedure, 1898', ['crpc:497']],
  ['section\n497 Cr.P.C.', ['crpc:497']],
  ['Section 489-F PPC and section 497 Cr.P.C.', ['ppc:489-F', 'crpc:497']],
  ['Article 199 of the Constitution', ['const:199']],
  ['section 510 Cr.P.C.', []],          // not in this test's statute set: left as plain text
  ['section 497 of the Act', []],       // no law named: not linked
  ['Section 12(2) of the CPC', ['cpc:12']],
  ['section 115, C. P. C.', ['cpc:115']],
  ['section 497 Cr.P.C. and Section 115 C.P.C', ['crpc:497', 'cpc:115']],   // neither code takes the other
  ['section 25 of the Guardians and Wards Act, 1890', ['gwa:25']],
  ['section 5(1) of the Family Courts Act, 1964', ['fca:5']],
  ['section 7 of the Muslim Family Laws Ordinance, 1961', ['mflo:7']],
  ['section 2 of the Dissolution of Muslim Marriages Act', ['dmma:2']],
];
let fails = 0;
for (const [text, want] of cases) {
  const got = keys(text);
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) fails++;
  console.log((ok ? 'ok   ' : 'FAIL ') + JSON.stringify(text) + ' -> ' + JSON.stringify(got));
}
if (statuteLaw('crpc:497') !== 'CrPC' || statuteLaw('ppc:302') !== 'PPC' || statuteLaw('const:9') !== 'Constitution' ||
    statuteLaw('gwa:25') !== 'Guardians and Wards Act') {
  fails++; console.log('FAIL statuteLaw');
}
// Case citations: one key per reported case, whichever form the judgment wrote.
var escapeHtml = x => String(x);
eval((['CITE_RE', 'COURTS'].map(line).join('\n') + '\n' + ['citeKey', 'indexCitations', 'linkCitations'].map(grab).join('\n'))
     .split('const ').join('var '));
const keyOf = t => { const m = [...t.matchAll(CITE_RE)]; return m.length ? citeKey(m[0]) : null; };
const same = [['PLD 1989 SC 166', '1989 SC 166'], ['PLJ 2014 Lahore 193', 'PLJ 2014 Lah. 193'],
              ['2015 P.Cr.L.J 7', '2015 PCrLJ 7'], ['Ghulam Haider v. Fateh Muhammad (2005 MLD 1501)', '2005 MLD 1501']];
same.forEach(([a, b]) => {
  const ok = keyOf(a) && keyOf(a) === keyOf(b);
  if (!ok) fails++;
  console.log((ok ? 'ok   ' : 'FAIL ') + a + ' == ' + b + ' -> ' + keyOf(a) + ' / ' + keyOf(b));
});
if (keyOf('2018 SCMR 2098') !== '2018 SCMR 2098' || keyOf('Sarfraz v. Khizer Hayat') !== null) { fails++; console.log('FAIL citeKey'); }
CITES = indexCitations([{doc_id: 'a', paragraph: '4', citation: 'PLD 1989 SC 166'},
                        {doc_id: 'b', paragraph: '7', citation: '1989 SC 166'},
                        {doc_id: 'b', paragraph: '7', citation: '2018 SCMR 2098'}]);
const c = CITES['1989 SC 166'];
if (!c || c.display !== 'PLD 1989 SC 166' || JSON.stringify(c.refs) !== '{"a":["4"],"b":["7"]}') { fails++; console.log('FAIL index ' + JSON.stringify(c)); }
const linked = linkCitations('relied on 1989 SC 166 and 2019 SCMR 1');
if (!linked.includes('data-cite="1989 SC 166"') || linked.includes('2019 SCMR 1"')) { fails++; console.log('FAIL link ' + linked); }
process.exit(fails ? 1 : 0);
