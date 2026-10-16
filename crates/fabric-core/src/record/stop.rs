// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Why an invocation stopped: the end line's `stop.cause`, in a fixed order
//! where the first match wins. A cause never changes `status`.

use super::lines::StopCause;
use crate::runtime::RunStatus;

/// What core knows about how an invocation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StopFacts {
    /// Final status.
    pub status: RunStatus,
    /// Core or the adapter host raised an error other than a timeout or a
    /// host exit, and the error did not come from the exchange with the host
    /// during an out-of-memory rise.
    pub fabric_failed: bool,
    /// The container's `oom_kill` count rose and the run owns its container.
    pub oom_rose: bool,
    /// Fabric's timeout fired, or the adapter stopped at the deadline core gave it.
    pub timed_out: bool,
    /// The adapter host exited before it answered.
    pub host_exited: bool,
    /// The adapter returned a failed result.
    pub agent_error: bool,
}

/// The stop cause for `facts`: none on success and on a cancel alone.
pub(crate) fn stop_cause(facts: StopFacts) -> Option<StopCause> {
    if facts.status == RunStatus::Succeeded {
        return None;
    }
    if facts.fabric_failed {
        return Some(StopCause::FabricFailed);
    }
    if facts.oom_rose {
        return Some(StopCause::SandboxFailed);
    }
    if facts.timed_out {
        return Some(StopCause::AgentExhaustedBudget);
    }
    if facts.status == RunStatus::Cancelled {
        return None;
    }
    if facts.host_exited {
        return Some(StopCause::Unclassified);
    }
    if facts.agent_error {
        return Some(StopCause::AgentFailed);
    }
    Some(StopCause::Unclassified)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failed() -> StopFacts {
        StopFacts {
            status: RunStatus::Failed,
            fabric_failed: false,
            oom_rose: false,
            timed_out: false,
            host_exited: false,
            agent_error: false,
        }
    }

    #[test]
    fn stop_cause_succeeded_run_has_no_cause_even_after_an_oom_rise() {
        let facts = StopFacts {
            status: RunStatus::Succeeded,
            oom_rose: true,
            ..failed()
        };

        assert_eq!(stop_cause(facts), None);
    }

    #[test]
    fn stop_cause_core_error_during_an_oom_kill_reads_fabric_failed() {
        let facts = StopFacts {
            fabric_failed: true,
            oom_rose: true,
            host_exited: true,
            ..failed()
        };

        assert_eq!(stop_cause(facts), Some(StopCause::FabricFailed));
    }

    #[test]
    fn stop_cause_oom_rise_with_a_host_exit_reads_sandbox_failed() {
        let facts = StopFacts {
            oom_rose: true,
            host_exited: true,
            timed_out: true,
            ..failed()
        };

        assert_eq!(stop_cause(facts), Some(StopCause::SandboxFailed));
    }

    #[test]
    fn stop_cause_timeout_reads_agent_exhausted_budget() {
        let facts = StopFacts {
            timed_out: true,
            agent_error: true,
            ..failed()
        };

        assert_eq!(stop_cause(facts), Some(StopCause::AgentExhaustedBudget));
    }

    #[test]
    fn stop_cause_cancel_alone_has_no_cause_but_an_oom_rise_still_counts() {
        let cancelled = StopFacts {
            status: RunStatus::Cancelled,
            ..failed()
        };

        assert_eq!(stop_cause(cancelled), None);
        assert_eq!(
            stop_cause(StopFacts {
                oom_rose: true,
                ..cancelled
            }),
            Some(StopCause::SandboxFailed)
        );
    }

    #[test]
    fn stop_cause_host_exit_reads_unclassified_before_an_agent_error() {
        let facts = StopFacts {
            host_exited: true,
            agent_error: true,
            ..failed()
        };

        assert_eq!(stop_cause(facts), Some(StopCause::Unclassified));
    }

    #[test]
    fn stop_cause_agent_error_reads_agent_failed_and_nothing_reads_unclassified() {
        assert_eq!(
            stop_cause(StopFacts {
                agent_error: true,
                ..failed()
            }),
            Some(StopCause::AgentFailed)
        );
        assert_eq!(stop_cause(failed()), Some(StopCause::Unclassified));
    }
}
