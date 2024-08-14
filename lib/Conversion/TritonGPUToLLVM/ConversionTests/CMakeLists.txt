add_mlir_translation_library(ConvertLayoutTestsLib
  ConvertLayoutTestsLib.cpp
  ConvertLayoutTestsLib.h
  ConvertLayoutTests.cpp

  LINK_COMPONENTS
  Core
  ExecutionEngine
  IRReader
  Interpreter
  MC
  Object

  DEPENDS
  LLVMIRIncGen

  LINK_LIBS
  ${CMAKE_DL_LIBS}
  PUBLIC
  MLIRArithToLLVM
  MLIRBuiltinToLLVMIRTranslation
  MLIRIndexToLLVM
  MLIRIR
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTargetLLVMIRExport
  TritonGPUToLLVM
  )

add_executable(ConvertLayoutTests ConvertLayoutTests.cpp)
target_link_libraries(ConvertLayoutTests PRIVATE GTest::gtest_main ConvertLayoutTestsLib)
target_compile_options(ConvertLayoutTests PRIVATE -fno-rtti)

include(GoogleTest)
gtest_discover_tests(ConvertLayoutTests)