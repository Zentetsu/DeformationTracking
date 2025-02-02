#include "MyComponent.h"

using sofa::helper::ReadAccessor;
using sofa::helper::WriteOnlyAccessor;

MyComponent::MyComponent() : d_myparam(initData(&d_myparam, 0.0f, "myparam", "This is a parameter.")) {}

MyComponent::~MyComponent() {}

int MyComponentClass = sofa::core::RegisterObject("This component does nothing.").add<MyComponent>();
