document.addEventListener('DOMContentLoaded', function () {
    const container = document.getElementById('strategies');
    const results = document.getElementById('results');
    const form = document.getElementById('btForm');

    // Build checkboxes for available strategies
    (window.available || []).forEach(name => {
        const div = document.createElement('div');
        div.className = 'strategy';
        const id = `chk_${name}`;
        div.innerHTML = `<label><input type="checkbox" id="${id}" /> <strong>${name}</strong></label>` +
            `<div>Params (JSON): <input id="params_${name}" style="width:80%" placeholder='{"n1":10}'/></div>`;
        container.appendChild(div);
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const symbol = document.getElementById('symbol').value;
        const start = document.getElementById('start').value;
        const end = document.getElementById('end').value;
        const strategies = {};
        (window.available || []).forEach(name => {
            const chk = document.getElementById(`chk_${name}`);
            if (chk && chk.checked) {
                const raw = document.getElementById(`params_${name}`).value || '{}';
                try { strategies[name] = JSON.parse(raw); } catch (e) { strategies[name] = {}; }
            }
        });
        if (Object.keys(strategies).length === 0) { alert('Select at least one strategy'); return; }

        results.textContent = 'Running...';
        const payload = { symbol, start: start || null, end: end || null, strategies };
        try {
            const r = await fetch('/run-backtest', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            const data = await r.json();
            results.textContent = JSON.stringify(data, null, 2);
        } catch (err) { results.textContent = String(err); }
    });
});