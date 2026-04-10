const fs = require('fs');
try {
    const runtime = require('@copilotkit/runtime');
    fs.writeFileSync('exports.txt', JSON.stringify(Object.keys(runtime), null, 2));
    console.log('Successfully wrote exports.txt');
} catch (e) {
    fs.writeFileSync('exports.txt', 'Error: ' + e.message);
    console.error(e);
}
