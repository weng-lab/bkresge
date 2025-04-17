library("BayesSpace")
library("ggplot2")

load(file = "/zata/zippy/kresgeb/psych_screen/output/bayes_space/jitter/chain_0.8.Rdata", verbose = TRUE)

ychange_vals <- as.numeric(chain)

plot <- plot(1:2500, ychange_vals,
    type = "l", lwd = 2,
    xlab = "Rep",
    ylab = "Ychange",
    main = "Traceplot of Ychange"
)

ggsave("/zata/zippy/kresgeb/psych_screen/output/blah.jpg", plot, width = 8, height = 6, dpi = 300)
