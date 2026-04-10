import { HttpAgent } from "@ag-ui/client";
import { AbstractAgent, AgentState } from "@copilotkit/runtime";

export class CopilotHttpAgent extends AbstractAgent<any> {
    private httpAgent: HttpAgent;

    public subscribers: any[] = [];
    public isRunning: boolean = false;
    public middlewares: any[] = [];
    public maxVersion: number = 1;

    constructor(httpAgent: HttpAgent) {
        super();
        this.httpAgent = httpAgent;
    }

    async execute(state: AgentState): Promise<AgentState> {
        const result = await this.httpAgent.run(state);
        return { ...state, ...result };
    }
}
