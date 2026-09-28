local dap = require('dap')

dap.configurations.python = {
    {
        name = "Launch Python File with Arguments",
        type = "python",
        request = "launch",
        program = "${file}",
        cwd = vim.fn.getcwd(),
        -- Enter the path to the python executable, if automatically finding does not work.
        pythonPath = vim.fn.exepath("python"),

        -- Prevent double prompts by ensuring args are defined only once
        args = function()
            local user_input = vim.fn.input("Enter script arguments (leave empty for none): ")
            if user_input == nil or user_input == "" then
                return {} -- Avoid passing empty args
            end
            return vim.split(user_input, " ") -- Properly split arguments
        end,
    }
}
