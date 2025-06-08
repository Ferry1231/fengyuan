[toc]
# Flow Matching Principles

This article mainly introduces the background and mathematical principles of Flow Matching. As the author is from an artificial intelligence background rather than mathematics, this article focuses on learning and understanding Flow Matching from the perspective of a deep learning practitioner. Many descriptions may lack rigorous mathematical proofs and explanations, and I ask for readers' understanding.

(References are temporarily omitted)

NFs, CNFs, and Flow Matching are flow-based techniques primarily used in generative models. The main task of generative models is to derive complex unknown distributions from known simple distributions, such as obtaining the distribution of an image dataset from standard Gaussian noise. The development of flow-based techniques involves continuously improving the tools of velocity fields and flows to better accomplish the task of fitting/transforming distributions. The essence of flow-based methods is to solve for the probability distribution of target points.

## 0. Vector Fields, Velocity Fields, and Flows

**Vector Field** $V(x)$ is a vector field defined in space, representing the direction and magnitude at each point $x$.

**Velocity Field** $v(x, t)$ is a special type of vector field, indicating the rate of change and tangential direction at each point $x$. Similar to the concept of velocity in physics, in deep learning, the velocity field is also used to represent "displacement"—the rate of change of data points.

In mathematical physics, a **flow** refers to a continuous change process evolving over time, such as the evolution of a dynamic system. In generative modeling, a flow describes the transformation process of data points, embodying the evolution of their distribution. For example, in CNFs (Continuous Normalizing Flows, introduced later), a flow can be expressed as follows:

Given a velocity field $v(x, t)$, the displacement $x = x(t)$, the evolution process can be described by the following partial differential equation:

$$
\frac{\partial{x(t)}}{\partial t} = v(x(t), t)
$$

Let $X_0 = x(0)$, then the flow is $\Phi^t(X_0) = x(t)$, which describes the process of evolving from $t=0$ to the current moment, i.e., the evolution of its probability distribution.

In NFs (Normalizing Flows, introduced later), although the transformation of the distribution is discrete, it essentially still represents the change process of the probability distribution, so it is also called a flow.

## 1. NFs: Normalizing Flows

NFs (Normalizing Flows) are a probability modeling method based on invertible transformations. Through a series of invertible transformations, a simple distribution (e.g., standard Gaussian distribution) can be transformed into a complex target distribution while maintaining a strict probability computation process. The mathematical form is as follows:

Given a distribution $z_0 \sim p(z_0)$,

and a series of invertible transformations: ${f_1, f_2, ..., f_k}$,

we have $z_i = f_i(z_{i-1}), p(z_i) = p(z_{i-1})|\det \frac{\partial f_i}{\partial z_{i-1}} |^{-1}$. After $K$ transformations, the target probability distribution can be obtained as follows:

$$
\log p(z_k) = \log p(z_0) - \Sigma_{i=1}^K \log |\det \frac{\partial f_i}{\partial z_{i-1}} |
$$

This is the solution for the target distribution.

![NFs Transformation Process](images/5v2-519b7d06728cb1c2dfff6153ef37b9b7.jpg)

**Main Drawbacks of Normalizing Flows**

1. All transformations $f$ must be invertible. First, common neural networks (e.g., ResNet, Transformer) are not necessarily invertible, making implementation challenging. Second, the invertibility of the model limits its expressive power, i.e., it struggles to effectively fit high-dimensional data distributions.
2. The number of network transformations is finite, limiting expressive power, with fixed sampling paths and low sampling efficiency.
3. The computational cost is high, as each transformation requires calculating the Jacobian determinant.

## 2. CNFs: Continuous Normalizing Flows

CNFs (Continuous Normalizing Flows) extend the discrete transformations in Normalizing Flows to a continuous scenario. In traditional NFs, the overall distribution transformation is achieved through a series of discrete invertible transformations $f$. CNFs introduce a continuous transformation process, making the transformation smoother and enhancing the model's expressive power.

CNFs define a continuous-time dynamic system, using an Ordinary Differential Equation (ODE) to describe the overall change:

Assume the initial distribution is $z_0 \sim p(z_0)$, then the ODE describing the change process is as follows:

$$
\frac{dz(t)}{dt} = f(z(t), t)
$$

where $f$ is a given vector field (i.e., velocity field), describing the rate and direction of change of $z$ at each moment.

Similar to NFs, the Jacobian determinant is needed to compute the probability change process. According to the Liouville theorem, the change in probability density $p(z(t))$ can be described by the following equation:

$$
\frac{\partial p(z(t))}{\partial t} = - \nabla (p(z(t))f(z(t), t))
$$

The rate of change is related to the divergence of the vector field. This equation can be solved using numerical methods for ODEs and PDEs (e.g., Euler method, Runge-Kutta method, etc.).

![From the original paper: https://arxiv.org/abs/1806.07366](images/6v2-47d1a64181b4cad4420dd375fb9b495e.jpg)

## 3. Flow Matching

FM (Flow Matching) is a method based on CNFs, used to train invertible generative models. In CNFs, directly solving the trajectory change $x(t)$ or the change in probability density $p(z(t))$ is quite challenging.

Given an initial state $X_0$ and its velocity field, the probability path is also uniquely determined. Thus, the core idea of FM is not to directly learn the change trajectory of data points but to learn the velocity field $v(x, t)$, allowing data points to move along the learned velocity field, transforming from $p_0(x)$ to $p_T(x)$. The sampling process can be considered as a transformation in the opposite direction of the velocity field, i.e., using the velocity field $-v(x, t)$ to transform from $p_T(x)$ to $p_0(x)$, completing the generation task.

The objective function of FM is very concise, minimizing the following loss:

$$
\mathcal{L}_{FM} = \mathbb E_{x_0, x_T, t} [||v_{\theta}(x_t, t) - v(x_t, t)||^2]
$$

where $x_t$ is an interpolated data point between $x_0$ and $x_T$, $v_{\theta}(x_t, t)$ is the learned velocity field, and $v(x_t, t)$ is the true interpolated velocity field, which can be derived from the interpolation path (details omitted here).

![](images/7v2-f68b4326ce380d8eec5d2f1427baa757.jpg)