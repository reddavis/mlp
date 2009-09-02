require File.dirname(__FILE__) + '/neuron'

class MLP
  
  def initialize(options={})
    @input_size = options[:inputs]
    @hidden_layers = options[:hidden_layers]
    @number_of_output_nodes = options[:output_nodes]
    setup_network
  end
  
  def feed_forward(input)
    @network.each_with_index do |layer, layer_index|
      layer.each do |neuron|
        if layer_index == 0
          neuron.fire(input)
        else
          input = @network[layer_index-1].map {|x| x.last_output}
          neuron.fire(input)
        end
      end
    end
    @network.last.map {|x| x.last_output}
  end
  
  def train(input, targets)
    # To go back we must go forward
    feed_forward(input)
    compute_deltas(targets)
    update_weights(input)
    calculate_error(targets)
  end
  
  def inspect
    @network
  end
  
  private
  
  def update_weights(input)
    reversed_network = @network.reverse
    reversed_network.each_with_index do |layer, layer_index|
      if layer_index == 0
        update_output_weights(layer, layer_index, input)
      else
        update_hidden_weights(layer, layer_index, input)
      end
    end
  end
  
  def update_output_weights(layer, layer_index, input)
    inputs = @hidden_layers.empty? ? input : @network[-2].map {|x| x.last_output}
    layer.each do |neuron|
      neuron.update_weight(inputs, 0.25)
    end
  end
  
  def update_hidden_weights(layer, layer_index, original_input)
    if layer_index == (@network.size - 1)
      inputs = original_input
    else
      inputs = @network.reverse[layer_index+1].map {|x| x.last_output}
    end
    layer.each do |neuron|
      neuron.update_weight(inputs, 0.25)
    end
  end
  
  def compute_deltas(targets)
    reversed_network = @network.reverse
    reversed_network.each_with_index do |layer, layer_index|
      if layer_index == 0
        compute_output_deltas(layer, targets)
      else
        compute_hidden_deltas(layer, targets)
      end
    end
  end
  
  def compute_output_deltas(layer, targets)
    layer.each_with_index do |neuron, i|
      output = neuron.last_output
      neuron.delta = output * (1 - output) * (targets[i] - output)
    end
  end
  
  def compute_hidden_deltas(layer, targets)
    layer.each_with_index do |neuron, neuron_index|
      error = 0
      @network.last.each do |output_neuron|
        error += output_neuron.delta * output_neuron.weights[neuron_index]
      end
      output = neuron.last_output
      neuron.delta = output * (1 - output) * error
    end
  end
  
  def calculate_error(targets)
    outputs = @network.last.map {|x| x.last_output}
    sum = 0
    targets.each_with_index do |t, index|
      sum += (t - outputs[index]) ** 2
    end
    0.5 * sum
  end
  
  def setup_network
    @network = []
    # Hidden Layers
    @hidden_layers.each_with_index do |number_of_neurons, index|
      layer = []
      inputs = index == 0 ? @input_size : @hidden_layers[index-1].size
      number_of_neurons.times { layer << Neuron.new(inputs) }
      @network << layer
    end
    # Output layer
    inputs = @hidden_layers.empty? ? @input_size : @hidden_layers.last
    layer = []
    @number_of_output_nodes.times { layer << Neuron.new(inputs)}
    @network << layer
  end
  
end