# Undersatnding GraphRegNet

This branch is to run and understand the parts of GraphRegNet.

## A big picture
This is a two stage process:
1. Using `Elastix` to do the initial alignment.
1. Use `GraphRegNet` to refine the local regions.

![img](rsc/workflow.PNG)

Source Draw.io file: `rsc\workflow.drawio` (can be edited by Draw.io Integration plugin in VS Code)

## Parts that can be replaced
`GraphRegNet` is designed for the lung. To adapt to other organs, different options can be tried for
* The MIND feature: perhaps some pre-trained CNN filters as a feature extractor
* Loss: can add a term to match the boundary
* Search grid: currently it's 15 x 15 x 15. If the organ is very small, then the search grid can be smaller.

## Test run
A test run using a simplie example is done in [Colab](https://colab.research.google.com/drive/1zWPZdNTqcCbjF2BjF63bU4-WtWnIukZh?usp=sharing).

Data and other outputs can be found in `exp`.

### Findings
* Loss is decreasing with epoch.
* The deformation field converges during 300 - 400 epochs. 
* The lung areas are mostly aligned.
* Some boundaries are not well aligned. 