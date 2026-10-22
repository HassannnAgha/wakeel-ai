const fs = require('fs');
const js = (() => { const src = fs.readFileSync(require('path').join(__dirname, '..', 'ui.py'), 'utf8'); const page = src.split('PAGE = r"""')[1].split('"""')[0]; return page.slice(page.indexOf('<script>') + 8, page.lastIndexOf('</script>')); })();
const grab = name => { const s = js.indexOf('function ' + name + '('); let d = 0;
  for (let j = js.indexOf('{', s); j < js.length; j++) { if (js[j] === '{') d++; else if (js[j] === '}') { d--; if (!d) return js.slice(s, j + 1); } } };
const consts = js.slice(js.indexOf('const fadd'), js.indexOf('const fstr')) + js.slice(js.indexOf('const fstr'), js.indexOf('\n', js.indexOf('const fstr')));
const lim = js.slice(js.indexOf('const LIMITATION'), js.indexOf('const UNIT'));
eval((lim + ['gcd', 'F', 'inheritance', 'addPeriod', 'limitationDeadline'].map(grab).join('\n') + consts).split('const ').join('var '));
const base = {deceased: 'male', husband: 0, wives: 0, sons: 0, daughters: 0, father: false, mother: false, brothers: 0, sisters: 0};
const run = h => Object.fromEntries(inheritance({...base, ...h}).shares.map(x => [x.key, fstr(x.share)]));
const cases = [
  ['wife + son + daughter', {wives: 1, sons: 1, daughters: 1}, {wives: '1/8', sons: '7/12', daughters: '7/24'}],
  ['husband + 2 daughters + father + mother (awl)', {deceased: 'female', husband: 1, daughters: 2, father: true, mother: true}, {husband: '3/15'.replace('3/15', '1/5'), daughters: '8/15', father: '2/15', mother: '2/15'}],
  ['wife + father + mother (Umariyya)', {wives: 1, father: true, mother: true}, {wives: '1/4', mother: '1/4', father: '1/2'}],
  ['husband + father + mother (Umariyya)', {deceased: 'female', husband: 1, father: true, mother: true}, {husband: '1/2', mother: '1/6', father: '1/3'}],
  ['one daughter only (radd)', {daughters: 1}, {daughters: '1'}],
  ['husband + mother (radd to mother)', {deceased: 'female', husband: 1, mother: true}, {husband: '1/2', mother: '1/2'}],
  ['daughter + father', {daughters: 1, father: true}, {daughters: '1/2', father: '1/2'}],
  ['2 daughters + brother', {daughters: 2, brothers: 1}, {daughters: '2/3', brothers: '1/3'}],
  ['daughter + sister (sister residuary)', {daughters: 1, sisters: 1}, {daughters: '1/2', sisters: '1/2'}],
  ['wife + brother + sister', {wives: 1, brothers: 1, sisters: 1}, {wives: '1/4', brothers: '1/2', sisters: '1/4'}],
  ['son + father: brothers excluded', {sons: 1, father: true, brothers: 2}, {father: '1/6', sons: '5/6'}],
  ['2 wives + 2 sons', {wives: 2, sons: 2}, {wives: '1/8', sons: '7/8'}],
  ['husband + 2 sisters (awl)', {deceased: 'female', husband: 1, sisters: 2}, {husband: '3/7', sisters: '4/7'}],
  ['wife only', {wives: 1}, {wives: '1'}],
  ['mother + 2 brothers', {mother: true, brothers: 2}, {mother: '1/6', brothers: '5/6'}],
];
let fails = 0;
for (const [name, h, want] of cases) {
  const got = run(h);
  const ok = JSON.stringify(Object.keys(want).sort().map(k => [k, want[k]])) === JSON.stringify(Object.keys(got).sort().map(k => [k, got[k]]));
  if (!ok) fails++;
  console.log((ok ? 'PASS ' : 'FAIL ') + name + (ok ? '' : '  got ' + JSON.stringify(got) + ' want ' + JSON.stringify(want)));
}
const d = (iso, art) => limitationDeadline(iso, art).deadline.toISOString().slice(0, 10);
const lc = [['2026-01-01', 152, '2026-01-31'], ['2026-01-31', 157, '2026-07-31'], ['2025-08-31', 157, '2026-02-28'],
            ['2024-02-29', 57, '2027-02-28'], ['2026-03-10', 156, '2026-06-08'], ['2026-10-01', 113, '2029-10-01']];
for (const [s, a, want] of lc) {
  const got = new Date(limitationDeadline(s, a).deadline.getTime() - limitationDeadline(s, a).deadline.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
  const ok = got === want; if (!ok) fails++;
  console.log((ok ? 'PASS ' : 'FAIL ') + `Art.${a} from ${s} -> ${got}` + (ok ? '' : ' want ' + want));
}
process.exit(fails ? 1 : 0);
