# This test was taken from ai4r gem

# Author::    Sergio Fierens
# License::   MPL 1.1
# Project::   ai4r
# Url::       http://ai4r.rubyforge.org/
#
# You can redistribute it and/or modify it under the terms of 
# the Mozilla Public License version 1.1  as published by the 
# Mozilla Foundation at http://www.mozilla.org/MPL/MPL-1.1.txt

require File.dirname(__FILE__) + '/training_patterns'
require File.dirname(__FILE__) + '/patterns_with_noise'
require File.dirname(__FILE__) + '/patterns_with_base_noise'
require File.dirname(__FILE__) + '/../lib/mlp'
require 'benchmark'

times = Benchmark.measure do
  
    srand 1
  
    net = MLP.new(:hidden_layers => [2], :output_nodes => 3, :inputs => 256)
    
    tr_with_noise = TRIANGLE_WITH_NOISE.flatten.collect { |input| input.to_f / 5.0}
    sq_with_noise = SQUARE_WITH_NOISE.flatten.collect { |input| input.to_f / 5.0}
    cr_with_noise = CROSS_WITH_NOISE.flatten.collect { |input| input.to_f / 5.0}

    tr_with_base_noise = TRIANGLE_WITH_BASE_NOISE.flatten.collect { |input| input.to_f / 5.0}
    sq_with_base_noise = SQUARE_WITH_BASE_NOISE.flatten.collect { |input| input.to_f / 5.0}
    cr_with_base_noise = CROSS_WITH_BASE_NOISE.flatten.collect { |input| input.to_f / 5.0}
    
    puts "Training the network, please wait."
    101.times do |i|
      tr_input = TRIANGLE.flatten.collect { |input| input.to_f / 5.0}
      sq_input = SQUARE.flatten.collect { |input| input.to_f / 5.0}
      cr_input = CROSS.flatten.collect { |input| input.to_f / 5.0}
      
      error1 = net.train(tr_input, [1,0,0])
      error2 = net.train(sq_input, [0,1,0])
      error3 = net.train(cr_input, [0,0,1])
      puts "Error after iteration #{i}:\t#{error1} - #{error2} - #{error3}" if i%20 == 0
    end

    def result_label(result)
      if result[0] > result[1] && result[0] > result[2]
        "TRIANGLE"
      elsif result[1] > result[2] 
        "SQUARE"
      else    
        "CROSS"
      end
    end
    
    tr_input = TRIANGLE.flatten.collect { |input| input.to_f / 5.0}
    sq_input = SQUARE.flatten.collect { |input| input.to_f / 5.0}
    cr_input = CROSS.flatten.collect { |input| input.to_f / 5.0}

    puts "Training Examples"
    puts "#{net.feed_forward(tr_input).inspect} => #{result_label(net.feed_forward(tr_input))}"
    puts "#{net.feed_forward(sq_input).inspect} => #{result_label(net.feed_forward(sq_input))}"
    puts "#{net.feed_forward(cr_input).inspect} => #{result_label(net.feed_forward(cr_input))}"
    puts "Examples with noise"
    puts "#{net.feed_forward(tr_with_noise).inspect} => #{result_label(net.feed_forward(tr_with_noise))}"
    puts "#{net.feed_forward(sq_with_noise).inspect} => #{result_label(net.feed_forward(sq_with_noise))}"
    puts "#{net.feed_forward(cr_with_noise).inspect} => #{result_label(net.feed_forward(cr_with_noise))}"
    puts "Examples with base noise"
    puts "#{net.feed_forward(tr_with_base_noise).inspect} => #{result_label(net.feed_forward(tr_with_base_noise))}"
    puts "#{net.feed_forward(sq_with_base_noise).inspect} => #{result_label(net.feed_forward(sq_with_base_noise))}"
    puts "#{net.feed_forward(cr_with_base_noise).inspect} => #{result_label(net.feed_forward(cr_with_base_noise))}"
  
end

puts "Elapsed time: #{times}"