set(corpus "${BINARY_DIR}/smoke.txt")
set(model "${BINARY_DIR}/smoke.nyh")
file(WRITE "${corpus}" "الرياضيات منطق. mathematics is reasoning. الرياضيات منطق.\n")

execute_process(COMMAND "${NIYAH_EXE}" train --input "${corpus}" --model "${model}"
                RESULT_VARIABLE r1 OUTPUT_VARIABLE o1 ERROR_VARIABLE e1)
if(NOT r1 EQUAL 0)
    message(FATAL_ERROR "train failed: ${r1}\n${o1}\n${e1}")
endif()
if(NOT o1 MATCHES "status=ok")
    message(FATAL_ERROR "train missing status=ok: ${o1}")
endif()

execute_process(COMMAND "${NIYAH_EXE}" eval --input "${corpus}" --model "${model}"
                RESULT_VARIABLE r2 OUTPUT_VARIABLE o2 ERROR_VARIABLE e2)
if(NOT r2 EQUAL 0)
    message(FATAL_ERROR "eval failed: ${r2}\n${o2}\n${e2}")
endif()
if(NOT o2 MATCHES "bits_per_byte=")
    message(FATAL_ERROR "eval missing metric: ${o2}")
endif()

execute_process(COMMAND "${NIYAH_EXE}" inspect --model "${model}"
                RESULT_VARIABLE r3 OUTPUT_VARIABLE o3 ERROR_VARIABLE e3)
if(NOT r3 EQUAL 0 OR NOT o3 MATCHES "format=NIYAHBG1")
    message(FATAL_ERROR "inspect failed: ${r3}\n${o3}\n${e3}")
endif()
