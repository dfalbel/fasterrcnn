region_proposal_network <- function(num_anchors) {
  keras::keras_model_custom(model_fn = function(self) {

    self$conv <- layer_conv_2d(
      filters = 512,
      kernel_size = c(3,3),
      padding = "same",
      activation = "relu",
      kernel_initializer = "normal"
    )

    self$class_conv <- layer_conv_2d(
      filters = num_anchors,
      kernel_size = c(1,1),
      activation = "sigmoid",
      kernel_initializer = "uniform"
    )

    self$regr_conv <- layer_conv_2d(
      filters = num_anchors * 4,
      kernel_size = c(1,1),
      activation = "linear",
      kernel_initializer = "zero"
    )

    function(inputs, mask = NULL) {
      x <- self$conv(inputs)
      list(
        x_class = self$class_conv(x),
        x_regr = self$regr_conv(x)
      )
    }
  },
  name = "region_proposal"
  )
}
