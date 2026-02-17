using DocStringExtensions

@template TYPES = """
                  $(TYPEDEF)
                  $(TYPEDFIELDS)
                  $(DOCSTRING)
                  """

@template (FUNCTIONS, METHODS, MACROS) = """
                                         $(TYPEDSIGNATURES)
                                         $(DOCSTRING)
                                         """
