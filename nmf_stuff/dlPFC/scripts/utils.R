suppressPackageStartupMessages({
    library(here)
})

#---------------------------------------------------------------
# Utility: setup_log
#---------------------------------------------------------------
# Creates a logs directory if missing, and starts a sink-based log file
# Returns the absolute path to the log file for reference
setup_log <- function(prefix = "run") {
    dir.create(here("logs"), showWarnings = FALSE, recursive = TRUE)
    timestamp <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")
    log_file <- here("logs", paste0(prefix, "_", timestamp, ".log"))
    sink(log_file, append = FALSE, split = TRUE)
    options(width = 120)
    log_msg(sprintf("Log file created: %s", normalizePath(log_file, mustWork = FALSE)))
    return(log_file)
}

#---------------------------------------------------------------
# Utility: log_msg
#---------------------------------------------------------------
# Prints a timestamped message to both console and log
log_msg <- function(msg) {
    cat(sprintf("[%s] %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), msg))
    flush.console()
}

#---------------------------------------------------------------
# Utility: close_log
#---------------------------------------------------------------
# Safely close sink
close_log <- function() {
    tryCatch(
        {
            sink()
        },
        error = function(e) {
            message("No sink to close.")
        }
    )
}

#---------------------------------------------------------------
# Utility: load_and_rename
#---------------------------------------------------------------
# Loads R objects from a .RData/.rda file and renames them in the target environment
# Provides detailed messages when verbose = TRUE
load_and_rename <- function(path, new_names = NULL, envir = .GlobalEnv, overwrite = FALSE, verbose = FALSE) {
    if (!file.exists(path)) {
        stop(sprintf("File not found: %s", path))
    }

    if (verbose) message(sprintf("Loading objects from: %s", path))

    # Capture objects that already exist before loading
    pre_existing <- ls(envir = envir)

    # Load all objects directly into target environment
    obj_names <- load(path, envir = envir, verbose = verbose)
    n <- length(obj_names)

    if (verbose) message(sprintf("Found %d object(s) in file: %s", n, path))

    if (!is.null(new_names)) {
        if (length(new_names) != n) {
            stop(sprintf(
                "Length of 'new_names' (%d) does not match number of objects in %s (%d)",
                length(new_names), path, n
            ))
        }
    } else {
        new_names <- obj_names
    }

    for (i in seq_along(obj_names)) {
        orig <- obj_names[i]
        new <- new_names[i]
        obj <- get(orig, envir = envir)

        # Check if the target name existed BEFORE the load (not just after)
        existed_before <- new %in% pre_existing

        if (existed_before && !overwrite) {
            stop(sprintf("Object '%s' already existed in the target environment before loading.", new))
        }

        if (orig != new) {
            assign(new, obj, envir = envir)
            rm(list = orig, envir = envir)
            if (verbose) message(sprintf("Renamed object '%s' -> '%s'", orig, new))
        } else if (verbose) {
            message(sprintf("Keeping original name: '%s'", orig))
        }

        if (verbose) {
            message(sprintf(
                "Class: %s, Size: %.2f MB",
                paste(class(obj), collapse = "/"),
                object.size(obj) / 1024^2
            ))
        }
    }

    if (verbose) message("Finished loading objects.\n")
    invisible(mget(new_names, envir = envir))
}


#---------------------------------------------------------------
# Utility: snapshot_script
#---------------------------------------------------------------
# Copies the current script to the log for reproducibility
snapshot_script <- function(script_path) {
    if (!file.exists(script_path)) {
        warning(sprintf("Script file not found: %s", script_path))
        return(NULL)
    }
    script_name <- basename(script_path)
    log_msg(sprintf("======Snapshot of script: %s======", script_name))
    # Write the script content to the log using
    cat("\n----- Begin Script Snapshot -----\n")
    script_content <- readLines(script_path)
    cat(script_content, sep = "\n")
    cat("\n----- End Script Snapshot -----\n\n")
}
