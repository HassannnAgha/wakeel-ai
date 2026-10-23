// Checks the calendar helpers in ui.py: finding dates in an answer and writing them as a calendar file.
// Run:  node tests/calendar_test.js   (pytest runs it through tests/test_ui.py)
const fs = require('fs');
const NL = String.fromCharCode(10), CRLF = String.fromCharCode(13, 10), BS = String.fromCharCode(92);
const js = (() => { const src = fs.readFileSync(require('path').join(__dirname, '..', 'ui.py'), 'utf8'); const page = src.split('PAGE = r"""')[1].split('"""')[0]; return page.slice(page.indexOf('<script>') + 8, page.lastIndexOf('</script>')); })();
const grab = name => { const s = js.indexOf('function ' + name + '('); let d = 0;
  for (let j = js.indexOf('{', s); j < js.length; j++) { if (js[j] === '{') d++; else if (js[j] === '}') { d--; if (!d) return js.slice(s, j + 1); } } };
const months = js.slice(js.indexOf('const MONTHS'), js.indexOf(NL, js.indexOf('const MONTHS')));
eval(months.replace('const ', 'var ') + NL + ['monthOf', 'datesIn', 'icsEscape', 'icsFold', 'icsOf'].map(grab).join(NL));
const assert = require('assert');
const today = new Date(2026, 9, 23);
const answer = [
  '## Key dates',
  '- **Date of the FIR:** 03.03.2024 [S2]',
  '- The cheque was dishonoured on 1st March 2024.',
  '- Next date of hearing: 18 November 2026, before the Judicial Magistrate.',
  '| Event | Date |', '| Reply due | November 30, 2026 |',
  '- Filed on 2026-10-01; a 31.02.2025 typo is not a date.',
  '- Reply within 15 days of receipt.',
].join(NL);
const d = datesIn(answer, today);
assert.deepStrictEqual(d.map(x => x.iso), ['2024-03-01', '2024-03-03', '2026-10-01', '2026-11-18', '2026-11-30']);
assert.deepStrictEqual(d.map(x => x.upcoming), [false, false, false, true, true]);
assert.strictEqual(d[3].line, 'Next date of hearing: 18 November 2026, before the Judicial Magistrate.');
assert.strictEqual(d[1].line, 'Date of the FIR: 03.03.2024');            // markdown and citation markers removed
assert.strictEqual(d[4].line, 'Reply due · November 30, 2026');      // a table row reads as a line
assert.strictEqual(datesIn('No dates here, only 15 days.', today).length, 0);
const ics = icsOf(d.filter(x => x.upcoming), 'Explained: summons.pdf', [1, 7], new Date(Date.UTC(2026, 9, 23, 12, 0, 0)));
assert.ok(ics.startsWith('BEGIN:VCALENDAR' + CRLF + 'VERSION:2.0' + CRLF) && ics.endsWith('END:VCALENDAR' + CRLF));
assert.strictEqual(ics.split('BEGIN:VEVENT').length - 1, 2);
assert.ok(ics.includes('DTSTART;VALUE=DATE:20261118' + CRLF + 'DTEND;VALUE=DATE:20261119'));
assert.strictEqual(ics.split('TRIGGER:-P1D').length - 1, 2);
assert.strictEqual(ics.split('TRIGGER:-P7D').length - 1, 2);
assert.ok(ics.includes('Explained: summons.pdf'));
assert.ok(ics.split(CRLF).every(l => l.length <= 75));                  // folded, as the format requires
assert.strictEqual(icsEscape('a, b; c' + BS + 'd' + NL + 'e'), 'a' + BS + ', b' + BS + '; c' + BS + BS + 'd' + BS + 'ne');
console.log('calendar ok');
