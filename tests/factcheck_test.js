// Checks the citation fact check in ui.py: names, sections, numbers and quotations in a cited
// sentence must be in the cited passage. Run:  node tests/factcheck_test.js [demo_cache]
// With a folder of cached answers, it also prints what it would flag in each (for calibration).
const fs = require('fs'), path = require('path');
const js = (() => { const src = fs.readFileSync(path.join(__dirname, '..', 'ui.py'), 'utf8'); const page = src.split('PAGE = r"""')[1].split('"""')[0]; return page.slice(page.indexOf('<script>') + 8, page.lastIndexOf('</script>')); })();
const grab = name => { const s = js.indexOf('function ' + name + '('); let d = 0;
  for (let j = js.indexOf('{', s); j < js.length; j++) { if (js[j] === '{') d++; else if (js[j] === '}') { d--; if (!d) return js.slice(s, j + 1); } } };
const block = (from, to) => js.slice(js.indexOf(from), js.indexOf(to));
eval((block('const NUMBER_WORDS', 'function factText') +
      ['citedNumbers', 'squash', 'factText', 'sourceFacts', 'hasNumber', 'factsIn', 'checkFacts'].map(grab).join('\n')).split('const ').join('var '));

const src = (n, excerpt, extra) => Object.assign({n, excerpt, title: '', case_no: '', court: 'LAHORE HIGH COURT', parties: {}}, extra || {});
const S = [
  src(1, 'The petitioner was arrested under section 489-F PPC on 20.1.2023 for a cheque of Rs.1,500,000. ' +
         'Bail was refused as the offence carries ten years.', {parties: {appellant: 'Azeem-ud-Din', respondent: 'Feroze Khan'}}),
  src(2, 'The court held that "the law favours adjudication on merits" and granted relief to Rashida Bibi. The Commission was told.'),
];
let fails = 0;
const check = (name, answer, wantMissing) => {
  const r = checkFacts(answer, S), got = r.missing.map(m => m.label);
  const ok = JSON.stringify(got) === JSON.stringify(wantMissing);
  if (!ok) fails++;
  console.log((ok ? 'ok   ' : 'FAIL ') + name + ' -> missing ' + JSON.stringify(got) + ' of ' + r.checked);
  return r;
};
check('supported facts pass', 'Azeem-ud-Din was arrested under section 489-F PPC for a cheque of Rs. 1,500,000 in 2023 [S1].', []);
check('number words count', 'The offence carries 10 years [S1].', []);
check('wrong section is caught', 'He was charged under section 420 PPC [S1].', ['420']);
check('wrong amount is caught', 'The cheque was for Rs. 2,000,000 [S1].', ['2,000,000']);
check('invented name is caught', 'The court relied on Kamran Shah [S1].', ['Kamran Shah']);
check('exact quotation passes', 'It said "the law favours adjudication on merits" [S2].', []);
check('altered quotation is caught', 'It said "the law always favours technicalities" [S2].', ['\u201cthe law always favours technicalities\u201d']);
check('bold labels are formatting, not names', '- **Conclusion/Finding:** Granted relief to Rashida Bibi [S2].', []);
check('a numbered bold label is formatting', '2. **Procedural Timings, Notes:** Granted relief to Rashida Bibi [S2].', []);
check('a plain label is not a name', 'Procedural: the court granted relief to Rashida Bibi [S2].', []);
check('a sentence opener is not a name', 'Citing Azeem-ud-Din, the court cancelled bail [S1].', []);
check('a quotation matches despite punctuation', 'It said "the law, favours (adjudication) on merits." [S2].', []);
check('quotation marks pair in order', 'He said "ok" and then "the law favours adjudication on merits" [S2].', []);
check('a defined term found in another passage is not flagged', 'The Commission acted [S1].', []);
check('a single capitalised word is not checked', 'The Registrar acted [S1].', []);
check('an invented two-word name is flagged', 'The Registrar, Imran Qureshi, acted [S1].', ['Imran Qureshi']);
const r = check('fact from the wrong passage is located', 'Rashida Bibi was arrested [S1].', ['Rashida Bibi']);
if (JSON.stringify(r.missing[0].elsewhere) !== '[2]') { fails++; console.log('FAIL elsewhere ' + JSON.stringify(r.missing[0].elsewhere)); }
{ // the user's own facts, cited alongside a judgment, are not citation errors
  const said = 'Tariq Mehmood borrowed Rs. 2,000,000 from you [S1].';
  const without = checkFacts(said, S).missing.map(m => m.label);
  const withUser = checkFacts(said, S, 'I lent Rs. 2,000,000 to Tariq Mehmood').missing.map(m => m.label);
  const ok = JSON.stringify(without) === JSON.stringify(['2,000,000', 'Tariq Mehmood']) && withUser.length === 0;
  if (!ok) fails++;
  console.log((ok ? 'ok   ' : 'FAIL ') + "the user's own facts count as a source -> " + JSON.stringify([without, withUser]));
}
check('uncited sentences are not checked', 'Kamran Shah won in 1999.', []);
check('multi-citation uses both passages', 'Rashida Bibi and Azeem-ud-Din both appear [S1, S2].', []);

// Calibration over real cached answers: print what would be flagged.
const dir = process.argv[2];
if (dir) {
  let checked = 0, missing = 0, inOther = 0;
  for (const f of fs.readdirSync(dir).filter(f => f.endsWith('.json'))) {
    const d = JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8'));
    const res = Array.isArray(d) ? d[d.length - 1] : d;
    if (!res.sources || !res.sources.length) continue;
    const c = checkFacts(res.answer, res.sources);
    checked += c.checked; missing += c.missing.length; inOther += c.missing.filter(m => m.elsewhere.length).length;
    console.log(f.slice(0, 8), c.checked, 'checked', JSON.stringify(c.missing.map(m => [m.label, m.cited, m.elsewhere])));
  }
  console.log('total', checked, 'checked,', missing, 'flagged (' + inOther + ' found in a different passage than the one cited, ' +
              (missing - inOther) + ' in no passage at all)');
}
process.exit(fails ? 1 : 0);
