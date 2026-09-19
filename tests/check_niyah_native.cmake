if(NOT DEFINED NIYAH_CLI)
    message(FATAL_ERROR "NIYAH_CLI is required")
endif()

function(check_native_case name text expected)
    execute_process(
        COMMAND "${NIYAH_CLI}"
            native
            --text "${text}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )

    if(NOT result EQUAL 0)
        message(FATAL_ERROR
            "${name}: native command failed: "
            "rc=${result} stderr=${error}")
    endif()

    if(NOT output STREQUAL expected)
        message(FATAL_ERROR
            "${name}: output mismatch\n"
            "EXPECTED=[${expected}]\n"
            "ACTUAL=[${output}]")
    endif()
endfunction()

check_native_case(
    "inside"
    "Is 192.168.1.42 inside 192.168.1.0/24?"
    "route=NETWORK_IP_IN_CIDR\naddress=192.168.1.42\nnetwork=192.168.1.0/24\nmatch=true\n"
)

check_native_case(
    "outside"
    "Is 192.168.2.42 inside 192.168.1.0/24?"
    "route=NETWORK_IP_IN_CIDR\naddress=192.168.2.42\nnetwork=192.168.1.0/24\nmatch=false\n"
)

check_native_case(
    "none"
    "What is 2+2?"
    "route=NONE\n"
)

execute_process(
    COMMAND "${NIYAH_CLI}"
        native
    RESULT_VARIABLE missing_result
    OUTPUT_VARIABLE missing_output
    ERROR_VARIABLE missing_error
)

if(NOT missing_result EQUAL 2)
    message(FATAL_ERROR
        "missing --text returned unexpected rc: "
        "${missing_result}")
endif()

execute_process(
    COMMAND "${NIYAH_CLI}"
        native
        --text "hello"
        --text "again"
    RESULT_VARIABLE duplicate_result
    OUTPUT_VARIABLE duplicate_output
    ERROR_VARIABLE duplicate_error
)

if(NOT duplicate_result EQUAL 2)
    message(FATAL_ERROR
        "duplicate --text returned unexpected rc: "
        "${duplicate_result}")
endif()

execute_process(
    COMMAND "${NIYAH_CLI}"
        native
        --unknown "x"
    RESULT_VARIABLE unknown_result
    OUTPUT_VARIABLE unknown_output
    ERROR_VARIABLE unknown_error
)

if(NOT unknown_result EQUAL 2)
    message(FATAL_ERROR
        "unknown native option returned unexpected rc: "
        "${unknown_result}")
endif()

message("P9H_NATIVE_CLI_CONTRACT=PASS")
