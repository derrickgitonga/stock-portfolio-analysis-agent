const { LangGraphAgent } = require('@copilotkit/runtime');
try {
    const proto = Object.getPrototypeOf(LangGraphAgent);
    console.log('LangGraphAgent extends:', proto.name);
} catch (e) {
    console.error(e);
}
