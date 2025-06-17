function(embed_resource input output)
  get_filename_component(directory ${input} DIRECTORY)
  get_filename_component(file ${input} NAME)
  add_custom_command(
    OUTPUT ${output}
    WORKING_DIRECTORY ${directory}
    COMMAND
      ${CMAKE_OBJCOPY} -B i386 -I binary -O elf64-x86-64 ${file} ${output}
    DEPENDS ${input})
  set_source_files_properties(
    ${output}
    PROPERTIES
      EXTERNAL_OBJECT true
      GENERATED true
  )
endfunction()
