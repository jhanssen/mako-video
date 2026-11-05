# ctest -L cpu: no GPU and no model (what CI runs); -L video: the model against its references.
# MiniMax-H3 against the Diffusers implementation's tensors (MAKO_H3_REF, written by tools/dump_h3_ref.py).
add_executable(h3_ref_test h3_ref_test.cpp)
target_link_libraries(h3_ref_test PRIVATE mako_videogen glaze_headers)
foreach(m tokens schedule text dit latents vae audio encode)
    add_test(NAME h3_${m} COMMAND h3_ref_test "${MAKO_H3}" "${MAKO_H3_REF}" ${m})
    set_tests_properties(h3_${m} PROPERTIES LABELS video SKIP_RETURN_CODE 77)
endforeach()

# The geometry and schedules without a model.
add_executable(geometry_test geometry_test.cpp)
target_link_libraries(geometry_test PRIVATE mako_videogen)
add_test(NAME geometry COMMAND geometry_test)
set_tests_properties(geometry PROPERTIES LABELS cpu)
