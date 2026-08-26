#!/usr/bin/env bash
# SWEEP2 - DEWEY parallel pool
# The script can run DEWEY individual preprocesssing scripts: 
# @M.Alberti + Claude
set -uo pipefail   # no -e: kill -0 failing is expected, normal control flow — not an error

# ============================================================
# Configuration
# ============================================================
N_PPOOL=2                    # number of subjects processed in parallel
SCRIPT="project folder"       # <-- set to your actual project root
BASH_DIR="${SCRIPT}/bash"
LOG_DIR="${SCRIPT}/logs"
mkdir -p "$LOG_DIR"

VPS=(15 16 17 18 19 20 21 22 24 25 26 27 28 29 30 33 35 36 37 38 39 41 42 43) #Participants number
SESSIONS=(ses-01 ses-02) #Sessions * participants

declare -A ICONS=( [done]="✅" [failed]="❌" [missing]="⚠️" ) # Claude add it 

# ============================================================


subjid_for() {  # subjid_for <vp> <session>
    printf 'sub-%02d_%s' "$1" "$2"
}

run_job() {  # run_job <vp> <session>  — meant to be backgrounded; output goes to its own log
    local vp="$1" session="$2"
    local subjid; subjid=$(subjid_for "$vp" "$session")
    local bash_script="${BASH_DIR}/${subjid}_DKI_Coreg_smoothing.sh"   # <-- adjust to your naming
    bash "$bash_script" > "${LOG_DIR}/${subjid}.log" 2>&1
}

reap() {  # reap <pid> — collects a finished job's exit code, records + prints its result
    local pid="$1"
    wait "$pid"
    local exit_code=$?
    local info="${pid_map[$pid]}"
    local vp="${info%%|*}" session="${info##*|}"
    local subjid; subjid=$(subjid_for "$vp" "$session")

    if (( exit_code == 0 )); then
        results[$subjid]="done"
        echo "  ${ICONS[done]} done    ${subjid}"
    else
        results[$subjid]="failed"
        echo "  ${ICONS[failed]} failed  ${subjid} (exit ${exit_code}) — see ${LOG_DIR}/${subjid}.log"
    fi

    unset "pid_map[$pid]"
    (( running-- ))
}

cleanup() {  # kill any still-running children if the script itself is interrupted
    echo -e "\nInterrupted — killing running jobs..."
    for pid in "${!pid_map[@]}"; do
        kill "$pid" 2>/dev/null
    done
    exit 130
}
trap cleanup INT TERM

# The script acts exactely as the jupyter notebook block, it first collect the all the jobs then run them in parallel
# ============================================================
# Build job list
# ============================================================
jobs=()
for vp in "${VPS[@]}"; do
    for session in "${SESSIONS[@]}"; do
        jobs+=("${vp}|${session}")
    done
done

echo
echo " Starting | ${#jobs[@]} jobs | ${N_PPOOL} parallel"
echo

# ============================================================
# Run in parallel (capped at N_PPOOL)
# ============================================================
declare -A pid_map   # pid -> "vp|session"
declare -A results   # subjid -> status
running=0

for job in "${jobs[@]}"; do
    vp="${job%%|*}"
    session="${job##*|}"
    subjid=$(subjid_for "$vp" "$session")
    bash_script="${BASH_DIR}/${subjid}_DKI_Coreg_smoothing.sh"

    # skip immediately if the expected script is missing — no point spawning a process for it
    if [[ ! -f "$bash_script" ]]; then
        echo "  ${ICONS[missing]} missing ${subjid} (${bash_script} not found)"
        results[$subjid]="missing"
        continue
    fi

    run_job "$vp" "$session" &
    pid=$!
    pid_map[$pid]="${vp}|${session}"
    (( running++ ))

    # block here until we've dropped back below the cap, polling rather than
    # mixing `wait -n` with explicit `wait $pid` (mixing the two can make a
    # successful job get misreported as failed — see notes below)
    while (( running >= N_PPOOL )); do
        for check_pid in "${!pid_map[@]}"; do
            kill -0 "$check_pid" 2>/dev/null || reap "$check_pid"
        done
        (( running >= N_PPOOL )) && sleep 0.2
    done
done

# ── reap whatever's still running after the last job was launched ──
for pid in "${!pid_map[@]}"; do
    reap "$pid"
done

# ============================================================
# Summary
# ============================================================
echo
echo "========== SUMMARY =========="
for subjid in "${!results[@]}"; do
    status="${results[$subjid]}"
    echo "  ${ICONS[$status]:-?}  ${subjid}: ${status}"
done | sort
echo "=============================="
