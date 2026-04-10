const fs = require('fs');
const { HttpAgent } = require('@ag-ui/client');
try {
    const agent = new HttpAgent({ url: 'http://localhost' });
    const output = {
        keys: Object.keys(agent),
        proto: Object.getOwnPropertyNames(Object.getPrototypeOf(agent))
    };
    fs.writeFileSync('agent_info.json', JSON.stringify(output, null, 2));
} catch (e) {
    fs.writeFileSync('agent_info.json', JSON.stringify({ error: e.message }));
}
