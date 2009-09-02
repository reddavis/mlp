# This test was taken from ai4r gem

require File.dirname(__FILE__) + '/../lib/mlp'
require 'benchmark'

times = Benchmark.measure do

  srand 1
  
  a = MLP.new(:hidden_layers => [2], :output_nodes => 1, :inputs => 2)

  3001.times do |i|
    a.train([0,0], [0])
    a.train([0,1], [1])
    a.train([1,0], [1])
    error = a.train([1,1], [0])
    puts "Error after iteration #{i}:\t#{error}" if i%200 == 0
  end

  puts "Test data"
  puts "[0,0] = > #{a.feed_forward([0,0]).inspect}"
  puts "[0,1] = > #{a.feed_forward([0,1]).inspect}"
  puts "[1,0] = > #{a.feed_forward([1,0]).inspect}"
  puts "[1,1] = > #{a.feed_forward([1,1]).inspect}"
  
end

puts "Elapsed time: #{times}"