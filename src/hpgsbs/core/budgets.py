"""Resource refusal is separate from scientific candidate selection."""
class ResourceBudgetExceeded(RuntimeError):
    def __init__(self, message, diagnostics):
        self.diagnostics = dict(diagnostics)
        super().__init__(message + "; diagnostics=" + repr(self.diagnostics))


class SearchCancelled(RuntimeError):
    pass
