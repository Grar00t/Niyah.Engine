if("$ENV{NIYAH_CASPER_MODEL_PATH}" STREQUAL "" OR
   "$ENV{NIYAH_CASPER_MODEL_SHA256}" STREQUAL "")
    message("SKIP: external model is not configured")
    return()
endif()

execute_process(COMMAND "${TEST_EXECUTABLE}" RESULT_VARIABLE result)
if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "Valid external model failed: ${result}")
endif()

foreach(bad_hash IN ITEMS
    "0000000000000000000000000000000000000000000000000000000000000000"
    "invalid"
    "gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env
        "NIYAH_CASPER_MODEL_SHA256=${bad_hash}" "${TEST_EXECUTABLE}"
        RESULT_VARIABLE result)
    if(NOT "${result}" STREQUAL "1")
        message(FATAL_ERROR "Invalid hash must return 1, got: ${result}")
    endif()
endforeach()

execute_process(COMMAND "${CMAKE_COMMAND}" -E env
    --unset=NIYAH_CASPER_MODEL_PATH "${TEST_EXECUTABLE}"
    RESULT_VARIABLE result)
if(NOT "${result}" STREQUAL "77")
    message(FATAL_ERROR "Missing model must return skip code 77, got: ${result}")
endif()
message("External model identity regression checks passed")
