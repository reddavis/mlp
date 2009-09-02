require 'helper'

class TestNeuron < Test::Unit::TestCase
  
  should "contain 3 weights (including weight for bias node)" do
    a = Neuron.new(2)
    assert_equal 3, a.inspect.size
  end
  
  should "save its last output" do
    a = Neuron.new(2)
    a.fire([0,1])
    assert a.last_output
  end
  
end
